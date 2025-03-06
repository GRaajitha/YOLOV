import os
from yolox.exp import get_exp
import argparse
import torch
import cv2
from yolox.data.data_augment import preproc

def make_parser():
    parser = argparse.ArgumentParser("YOLOV onnx export parser")
    parser.add_argument(
        "-f",
        "--exp_file",
        default='',
        type=str,
        help="input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='', type=str, help="checkpoint file")
    parser.add_argument(
        "-o",
        "--out_file",
        default='',
        type=str,
        help="path for output onnx file",
    )
    parser.add_argument(
        "--path", default="", help="path to images or video"
    )
    parser.add_argument("--conf", default=0.05, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    return parser

if __name__ == "__main__":    
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # resume the model/optimizer state dict
    model.load_state_dict(ckpt["model"])
    if args.device == "gpu":
        model.cuda()
    model.eval()

    gframe = exp.gframe_val
    lframe = exp.lframe_val

    cap = cv2.VideoCapture(args.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(
        str(args.out_file),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input = preproc(frame, (1920, 1920))
        img_bgr = plot_detections(model, input)
        img_bgr = cv2.resize(img_bgr, (width, height))
        video_writer.write(img_bgr)

    video_writer.release()