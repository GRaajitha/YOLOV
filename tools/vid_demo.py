#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import gc
import cv2
from tqdm import tqdm
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets.vid_classes import VID_classes
#from yolox.data.datasets.vid_classes import OVIS_classes as VID_classes
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from val_to_imdb import Predictor
from yolox.models.post_process import post_linking
import random
import json
import REPP

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png",".JPEG"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOV Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="/mnt/weka/scratch/yuheng.shi/dataset/VID/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00130000.mp4", help="path to images or video"
    )

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='./exps/yolov++/v++_SwinBaseX_decoupleReg.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='./V++_outputs/v++_SwinBaseX_decoupleReg/best_ckpt.pth', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--dataset",
        default='vid',
        type = str,
        help = "Decide pred classes"
    )
    parser.add_argument("--conf", default=0.05, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--gframe', default=32, help='global frame num')
    parser.add_argument('--lframe', default=0, help='local frame num')
    parser.add_argument('--save_result', default=True)
    parser.add_argument('--post', default=False,action="store_true")
    parser.add_argument('--repp_cfg', default='./tools/yolo_repp_cfg.json' ,help='repp cfg filename', type=str)
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name,[image_name])
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def imageflow_demo(predictor, vis_folder, current_time, args, exp):
    gframe = exp.gframe_val
    lframe = exp.lframe_val
    traj_linking = exp.traj_linking
    P, Cls = exp.defualt_p, exp.num_classes

    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    os.makedirs(save_folder, exist_ok=True)
    ratio = min(predictor.test_size[0] / height, predictor.test_size[1] / width)
    ratio_y, ratio_x = predictor.test_size[0] / height, predictor.test_size[1] / width
    vid_save_path = os.path.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video save_path is {vid_save_path}")
    vid_writer = cv2.VideoWriter(
        vid_save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    # Process frames in sliding window
    window_size = gframe if gframe != 0 else lframe
    if window_size == 0:
        window_size = 32  # Default window size if both gframe and lframe are 0

    # Initialize sliding window
    frames_window = []
    frame_idx = 0
    processed_frames = 0

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        # Preprocess frame
        processed_frame, _ = predictor.preproc(frame, None, exp.test_size)
        frames_window.append((frame, torch.tensor(processed_frame)))

        # When window is full, process the batch
        if len(frames_window) == window_size:
            # Extract frames and tensors
            original_frames = [f[0] for f in frames_window]
            frame_tensors = [f[1] for f in frames_window]

            # Process the batch
            if traj_linking:
                batch_outputs, adj_lists, fc_outputs, names = predictor.inference(
                    torch.stack(frame_tensors), lframe=window_size, gframe=0
                )
            else:
                batch_outputs = predictor.inference(
                    torch.stack(frame_tensors), lframe=lframe, gframe=gframe
                )

            # Visualize and save the last frame of the window
            if args.post:
                ratio = 1
            result_frame = predictor.visual(
                batch_outputs[-1], original_frames[-1], ratio_y, ratio_x, cls_conf=args.conf
            )
            if args.save_result:
                vid_writer.write(result_frame)
            processed_frames += 1

            # Slide the window by removing the first frame
            frames_window.pop(0)

            # Clear memory
            del frame_tensors
            del batch_outputs
            gc.collect()
            torch.cuda.empty_cache()

        frame_idx += 1

    # Process remaining frames in the window
    if len(frames_window) > 1:
        original_frames = [f[0] for f in frames_window]
        frame_tensors = [f[1] for f in frames_window]

        if traj_linking:
            batch_outputs, adj_lists, fc_outputs, names = predictor.inference(
                torch.stack(frame_tensors), lframe=len(frame_tensors), gframe=0
            )
        else:
            batch_outputs = predictor.inference(
                torch.stack(frame_tensors), lframe=lframe, gframe=gframe
            )

        # Visualize and save the last frame
        if args.post:
            ratio = 1
        result_frame = predictor.visual(
            batch_outputs[-1], original_frames[-1], ratio_y, ratio_x, cls_conf=args.conf
        )
        if args.save_result:
            vid_writer.write(result_frame)
        processed_frames += 1

    # Clean up
    cap.release()
    vid_writer.release()

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(args.output_dir,file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    if args.dataset=='vid':
        repp_params = json.load(open(args.repp_cfg, 'r'))
        post = REPP.REPP(**repp_params)
        predictor = Predictor(model, exp, VID_classes, trt_file, decoder, args.device, args.legacy,post=post)
    else:
        predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.legacy)
    current_time = time.localtime()

    imageflow_demo(predictor, vis_folder, current_time, args,exp)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.traj_linking = True and exp.lmode
    exp.lframe_val = int(args.lframe)
    exp.gframe_val = int(args.gframe)
    main(exp, args)
