#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np
import cv2
import os
import wandb
import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.models.post_process import postpro_woclass,post_threhold
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    time_synchronized,
    synchronize,
    xyxy2xywh
)

def log_pr_curve(cocoEval, iou_threshold):
    """Logs the Precision-Recall curve averaged over categories to wandb.

    Args:
        cocoEval (COCOeval): The evaluated COCOeval object.
        iou_threshold (float): The IoU threshold to use for the curve.
    """
    if not wandb.run:
        logger.warning("WandB run not initialized. Skipping PR curve logging.")
        return
    
    # Find the index for the given IoU threshold. Will raise IndexError if not found.
    iou_idx = np.where(np.isclose(cocoEval.params.iouThrs, iou_threshold))[0][0]
    # Default indices for area range ('all') and max detections (100)
    area_idx = 0
    max_dets_idx = cocoEval.params.maxDets.index(100)

    # Extract precision values across all recall thresholds (R) and categories (K) Shape: [R, K]
    precisions_all_categories = cocoEval.eval['precision'][iou_idx, :, :, area_idx, max_dets_idx]
    # Average precision across categories (axis=1), treating -1 as NaN
    mean_precision = np.nanmean(np.where(precisions_all_categories == -1, np.nan, precisions_all_categories), axis=1)
    # Get recall thresholds
    recall = cocoEval.params.recThrs

    # Filter out NaN values from averaging and corresponding recall values
    valid_mask = ~np.isnan(mean_precision)
    final_precision = mean_precision[valid_mask]
    final_recall = recall[valid_mask]

    # Check if there's data to plot
    if final_recall.size == 0 or final_precision.size == 0:
        logger.warning(f"No valid data points found to log the PR curve for IoU={iou_threshold}.")
    else:
        # Create wandb.Table
        data = [[r, p] for r, p in zip(final_recall, final_precision)]
        table = wandb.Table(data=data, columns=["recall", "precision"])

        # Log custom line plot to wandb
        log_key = f"pr_curve_iou_{str(iou_threshold).replace('.', '_')}"
        title = f"Precision-Recall Curve (IoU={iou_threshold}, Averaged Over Categories)"
        wandb.log({log_key: wandb.plot.line(table, "recall", "precision", title=title)})
        logger.info(f"Logged PR curve for IoU={iou_threshold} to WandB under key '{log_key}'.")

def log_confusion_matrix(cocoEval, cat_names):
    """creates and logs confidence matrix and normalized confidence matrix tables in wandb"""
    if not wandb.run:
            logger.warning("WandB run not initialized. Skipping PR curve logging.")
            return
    
    conf_mat_data = []
    cat_names = cat_names + ['background']
    for i in range(len(cocoEval.conf_matrix)):
        conf_mat_data.append(
            [cat_names[i]] + list(cocoEval.conf_matrix[i])
        )

    wandb.log(
        {
            f"confusion_matrix@0_5_IOU": wandb.Table(
                columns=["GT/pred"] + list(cat_names),
                data=conf_mat_data,
            )
        }
    )

def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table, per_class_AR


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table, per_class_AP


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        max_epoch: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
        fg_AR_only: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.fg_AR_only = fg_AR_only
        self.max_epoch_id = max_epoch-1

    def evaluate(
        self,
        model,
        epoch=0,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                if not self.fg_AR_only:
                    outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre
                    )
                else:
                    outputs = post_threhold(
                        outputs, self.num_classes,
                    )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

                            #vizualize
            if cur_iter == 0:
                output_dir = "./YOLOX_Outputs/eval_viz"
                os.makedirs(output_dir, exist_ok=True)
                for i in range(imgs.shape[0]):
                    img = imgs[i]
                    img = img.cpu().detach().numpy()
                    img = img.astype('uint8')  # Convert to uint8

                    # Convert from (C, H, W) to (H, W, C)
                    img = img.transpose(1, 2, 0)
                    img = np.ascontiguousarray(img)

                    for j in range(outputs[i].shape[0]):
                        xmin, ymin, xmax, ymax, obj_score, cls_score, cls = outputs[i][j]
                        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                        score = obj_score * cls_score
                        score = round(score.item(), 3)
                        #if score > 0.001:
                        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
                        # img = cv2.putText(img, str(f"{cls.item()}_{score}"), (xmin+5, ymin+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                    # log the image
                    cv2.imwrite(f"{output_dir}/image_{i}.png", img)

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)
        #calculate average predictions per image
        if is_main_process():
            logger.info("average predictions per image: {:.2f}".format(len(data_list)/len(self.dataloader.dataset)))

        eval_results = self.evaluate_prediction(data_list, statistics, epoch)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                if self.fg_AR_only: # for testing the forgound class AR
                    label = 0
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics, epoch):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            # try:
            #     from yolox.layers import COCOeval_opt as COCOeval
            # except ImportError:
            #     from pycocotools.cocoeval import COCOeval

            #     logger.warning("Use standard COCOeval.")

            from tools.cocoeval_custom import COCOeval

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.params.iouThrs = np.array([0.2, 0.5, 0.75])
            cocoEval.params.useCats = 1  # Enable category-based evaluation
            cocoEval.evaluate()
            cocoEval.accumulate()

            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize(epoch == self.max_epoch_id)

            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if epoch == self.max_epoch_id:
                log_pr_curve(cocoEval, iou_threshold=0.5)
                log_confusion_matrix(cocoEval, cat_names)
            if self.per_class_AP:
                AP_table, per_class_AP = per_class_AP_table(cocoEval, class_names=cat_names)
                wandb.log({f"val/mAP_{name}":value/100 for name, value in per_class_AP.items()})
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table, per_class_AR = per_class_AR_table(cocoEval, class_names=cat_names)
                wandb.log({f"val/mAR_{name}":value/100 for name, value in per_class_AR.items()})
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
