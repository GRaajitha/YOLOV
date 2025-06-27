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


def evaluate_per_attribute_per_class(cocoGt, cocoDt, cat_names, attribute_names=None):
    """Evaluate detection performance per attribute per class.
    
    Args:
        cocoGt: COCO ground truth object
        cocoDt: COCO detection results object
        cat_names: List of category names
        attribute_names: List of attribute names to evaluate (if None, auto-detect)
    
    Returns:
        results: Dictionary with per-attribute per-class metrics
    """
    if attribute_names is None:
        # Auto-detect attributes from the first annotation that has attributes
        attribute_names = []
        for ann in cocoGt.anns.values():
            if 'attributes' in ann:
                attribute_names = list(ann['attributes'].keys())
                break
    
    if not attribute_names:
        logger.warning("No attributes found in annotations")
        return {}
    
    results = {}
    
    for attr_name in attribute_names:
        logger.info(f"Evaluating attribute: {attr_name}")
        
        # Get all unique values for this attribute
        attr_values = set()
        for ann in cocoGt.anns.values():
            if 'attributes' in ann and attr_name in ann['attributes']:
                attr_val = ann['attributes'][attr_name]
                # Handle None values
                if attr_val is None:
                    attr_values.add("None")
                elif isinstance(attr_val, list):
                    for val in attr_val:
                        if val is None:
                            attr_values.add("None")
                        else:
                            attr_values.add(str(val))
                else:
                    attr_values.add(str(attr_val))
        
        attr_results = {}
        
        for attr_val in attr_values:
            # Skip empty or invalid attribute values
            if not attr_val or attr_val.strip() == "":
                continue
            
            logger.info(f"Processing attribute value: {attr_val}")
            
            # Create filtered ground truth and detection sets for this attribute value
            filtered_gt_anns = []
            filtered_dt_anns = []
            
            # Filter ground truth annotations for this attribute value
            for ann in cocoGt.anns.values():
                # Skip ignored annotations
                if ann.get('is_ignore', 0) == 1:
                    continue
                    
                if 'attributes' in ann and attr_name in ann['attributes']:
                    ann_attr_val = ann['attributes'][attr_name]
                    # Handle None values
                    if ann_attr_val is None:
                        if attr_val == "None":
                            filtered_gt_anns.append(ann)
                    elif isinstance(ann_attr_val, list):
                        if str(attr_val) in [str(v) if v is not None else "None" for v in ann_attr_val]:
                            filtered_gt_anns.append(ann)
                    else:
                        if str(ann_attr_val) == str(attr_val):
                            filtered_gt_anns.append(ann)
            
            if not filtered_gt_anns:
                logger.info(f"No ground truth annotations found for attribute value: {attr_val}")
                continue
            
            # Get all image IDs that have ground truth with this attribute value
            gt_img_ids = set(ann['image_id'] for ann in filtered_gt_anns)
            
            # Filter detection annotations for images that have ground truth with this attribute value
            for dt_ann in cocoDt.anns.values():
                if dt_ann['image_id'] in gt_img_ids:
                    filtered_dt_anns.append(dt_ann)
            
            # If no detections, create a dummy detection to satisfy COCO format
            if not filtered_dt_anns:
                dummy_dt_ann = {
                    'image_id': list(gt_img_ids)[0],
                    'category_id': 0,  # Use first category
                    'bbox': [0, 0, 1, 1],
                    'score': 0.0
                }
                filtered_dt_anns = [dummy_dt_ann]
            
            # Create COCO objects for this attribute value
            try:
                from tools.cocoeval_custom import COCOeval
                
                # Handle images properly
                if hasattr(cocoGt, 'imgs') and isinstance(cocoGt.imgs, dict):
                    filtered_images = [cocoGt.imgs[img_id] for img_id in gt_img_ids if img_id in cocoGt.imgs]
                else:
                    filtered_images = []
                    for img in cocoGt.dataset.get('images', []):
                        if img.get('id') in gt_img_ids:
                            filtered_images.append(img)
                
                # Ensure category IDs are integers
                categories = []
                for cat in cocoGt.cats.values():
                    cat_copy = cat.copy()
                    cat_copy['id'] = int(cat_copy['id'])
                    categories.append(cat_copy)
                
                # Prepare ground truth annotations
                filtered_gt_anns_fixed = []
                for ann in filtered_gt_anns:
                    ann_copy = ann.copy()
                    ann_copy['category_id'] = int(ann_copy['category_id'])
                    ann_copy['image_id'] = int(ann_copy['image_id'])
                    ann_copy["clean_bbox"] = [int(val) for val in ann_copy["clean_bbox"]]
                    filtered_gt_anns_fixed.append(ann_copy)
                
                # Prepare detection annotations (only essential fields)
                filtered_dt_anns_fixed = []
                for ann in filtered_dt_anns:
                    ann_copy = {
                        'image_id': int(ann['image_id']),
                        'category_id': int(ann['category_id']),
                        'bbox': ann['bbox'],
                        'score': ann['score']
                    }
                    filtered_dt_anns_fixed.append(ann_copy)
                
                # Ensure image IDs are integers
                filtered_images_fixed = []
                for img in filtered_images:
                    img_copy = img.copy()
                    img_copy['id'] = int(img_copy['id'])
                    filtered_images_fixed.append(img_copy)
                
                # Create filtered COCO format
                filtered_gt = {
                    'annotations': filtered_gt_anns_fixed,
                    'images': filtered_images_fixed,
                    'categories': categories
                }
                
                # Validate that we have the required data
                if not filtered_gt_anns_fixed:
                    logger.warning(f"No ground truth annotations found for {attr_name}_{attr_val}")
                    continue
                
                if not filtered_images_fixed:
                    logger.warning(f"No images found for {attr_name}_{attr_val}")
                    continue
                
                if not categories:
                    logger.warning(f"No categories found for {attr_name}_{attr_val}")
                    continue
                
                # Create temporary files
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(filtered_gt, f)
                    gt_file = f.name
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(filtered_dt_anns_fixed, f)
                    dt_file = f.name
                
                # Create COCO objects and evaluate
                from pycocotools.coco import COCO
                temp_gt = COCO(gt_file)
                temp_dt = temp_gt.loadRes(dt_file)
                
                coco_eval = COCOeval(temp_gt, temp_dt, 'bbox')
                coco_eval.params.iouThrs = np.array([0.5])
                coco_eval.evaluate()
                coco_eval.accumulate()
                
                # Get per-class metrics by calling summarize for each class
                for cat_idx, cat_name in enumerate(cat_names):
                    # Count ground truth and detections for this class
                    gt_count = sum(1 for ann in filtered_gt_anns if ann['category_id'] == cat_idx)
                    dt_count = sum(1 for ann in filtered_dt_anns if ann['category_id'] == cat_idx)
                    
                    if gt_count > 0:  # Only include if there are ground truth instances
                        # Set the category ID for evaluation
                        coco_eval.params.catIds = [cat_idx]
                        
                        # Re-evaluate for this specific class
                        coco_eval.evaluate()
                        coco_eval.accumulate()
                        coco_eval.summarize(printSummary=False)
                        
                        # Extract metrics from stats
                        ap_all = coco_eval.stats[1] if len(coco_eval.stats) > 1 else 0.0  # AP @ IoU=0.50
                        ar_all = coco_eval.stats[8] if len(coco_eval.stats) > 8 else 0.0  # AR @ maxDets=100
                        ap_small = coco_eval.stats[3] if len(coco_eval.stats) > 3 else 0.0  # AP @ small
                        ar_small = coco_eval.stats[9] if len(coco_eval.stats) > 9 else 0.0  # AR @ small
                        ap_medium = coco_eval.stats[4] if len(coco_eval.stats) > 4 else 0.0  # AP @ medium
                        ar_medium = coco_eval.stats[10] if len(coco_eval.stats) > 10 else 0.0  # AR @ medium
                        ap_large = coco_eval.stats[5] if len(coco_eval.stats) > 5 else 0.0  # AP @ large
                        ar_large = coco_eval.stats[11] if len(coco_eval.stats) > 11 else 0.0  # AR @ large
                        
                        key = f"{cat_name}_{attr_name}_{attr_val}"
                        attr_results[key] = {
                            'class': cat_name,
                            'attribute': attr_name,
                            'value': attr_val,
                            f'ap@IoU={coco_eval.params.iouThrs}': ap_all,
                            'ar@maxDets=100': ar_all,
                            'ap_small<0.0123%': ap_small,
                            'ar_small<0.0123%': ar_small,
                            'ap_medium<0.111%': ap_medium,
                            'ar_medium<0.111%': ar_medium,
                            'ap_large>0.111%': ap_large,
                            'ar_large>0.111%': ar_large,
                            'num_gt': gt_count,
                            'num_dt': dt_count
                        }
                
                # Clean up temporary files
                os.unlink(gt_file)
                os.unlink(dt_file)
                
            except Exception as e:
                logger.warning(f"Error evaluating {attr_name}_{attr_val}: {e}")
                continue
        
        results[attr_name] = attr_results
    
    return results


def log_per_attribute_per_class_metrics(results, iouThrs=[0.5]):
    """Log per-attribute per-class metrics to wandb as tables.
    
    Args:
        results: Results from evaluate_per_attribute_per_class
        epoch: Current epoch number
    """
    
    for attr_name, attr_results in results.items():
        # Create table data for this attribute
        table_data = []
        table_columns = ["Class", "Attribute Value", "AP", "AR", "AP_small<0.0123%", "AR_small<0.0123%", "AP_medium<0.111%", "AR_medium<0.111%", "AP_large>0.111%", "AR_large>0.111%", "Ground Truth Count", "Detection Count"]
        
        for key, data in attr_results.items():
            if data['num_gt'] > 0:  # Only include if there are ground truth instances
                table_data.append([
                    data['class'],
                    data['value'],
                    round(data[f'ap@IoU={iouThrs}'], 3),
                    round(data['ar@maxDets=100'], 3),
                    round(data['ap_small<0.0123%'], 3),
                    round(data['ar_small<0.0123%'], 3),
                    round(data['ap_medium<0.111%'], 3),
                    round(data['ar_medium<0.111%'], 3),
                    round(data['ap_large>0.111%'], 3),
                    round(data['ar_large>0.111%'], 3),
                    data['num_gt'],
                    data['num_dt']
                ])
        
        if table_data:
            # Print a nicely formatted table
            from tabulate import tabulate
            print(f"\nPer-Attribute Per-Class Results for '{attr_name}':")
            print("=" * 80)
            print(tabulate(table_data, headers=table_columns, tablefmt="grid", floatfmt=".3f"))
            print("=" * 80)
            
            # Create WandB table
            table = wandb.Table(
                columns=table_columns,
                data=table_data
            )
            
            if not wandb.run:
                logger.warning("WandB run not initialized. Skipping attribute metrics logging.")
                continue
            # Log the table
            wandb.log({f"per_attribute_per_class_{attr_name}": table})
            
            # Also log a summary metric (average AP across all values for this attribute)
            valid_aps = [data[f'ap@IoU={iouThrs}'] for data in attr_results.values() if data['num_gt'] > 0]
            if valid_aps:
                avg_ap = sum(valid_aps) / len(valid_aps)
                wandb.log({f"val/avg_AP_{attr_name}": avg_ap})
            
            logger.info(f"Logged per-attribute per-class table for '{attr_name}' to WandB")
        else:
            logger.warning(f"No valid data to log for attribute '{attr_name}'")


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
        per_attribute_per_class: bool = False,
        attribute_names: list = None,
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
            per_attribute_per_class: Show per attribute per class metrics during evaluation. Default to False.
            attribute_names: List of attribute names to evaluate. If None, auto-detect from annotations.
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
        self.per_attribute_per_class = per_attribute_per_class
        self.attribute_names = attribute_names
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
            
            # Per-attribute per-class evaluation
            if self.per_attribute_per_class and epoch == self.max_epoch_id:
                logger.info("Starting per-attribute per-class evaluation...")
                try:
                    # Check if annotations have attributes
                    has_attributes = False
                    for ann in cocoGt.anns.values():
                        if 'attributes' in ann:
                            has_attributes = True
                            break

                    if not has_attributes:
                        logger.warning("No attributes found in annotations. Skipping per-attribute per-class evaluation.")
                    else:
                        # Perform per-attribute per-class evaluation
                        attr_results = evaluate_per_attribute_per_class(
                            cocoGt, cocoDt, cat_names, self.attribute_names
                        )
                        
                        if attr_results:
                            # Add summary to info string
                            info += "\nPer-Attribute Per-Class Results:\n"
                            log_per_attribute_per_class_metrics(attr_results)
                        
                except Exception as e:
                    logger.error(f"Error in per-attribute per-class evaluation: {e}")
                    info += f"\nError in per-attribute per-class evaluation: {e}\n"
            
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
