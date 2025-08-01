#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import copy
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm
from yolox.evaluators.coco_evaluator import (
    per_class_AR_table, 
    per_class_AP_table, 
    log_pr_curve, 
    log_confusion_matrix,
    evaluate_per_attribute_per_class,
    log_per_attribute_per_class_metrics
)
import torch
import pycocotools.coco
import os
import numpy as np
import cv2
import wandb
import matplotlib.pyplot as plt
from yolox.data.datasets.vid_classes import VID_classes

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
    get_rank
)

class VIDEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
            self, dataloader, img_size, confthre, nmsthre,
            num_classes, max_epoch, testdev=False, gl_mode=False,
            lframe=0, gframe=32, output_dir="./", per_class_AP=False,
            per_class_AR=False, per_attribute_per_class=False,
            attribute_names=None, **kwargs
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP (bool): Show per class AP during evaluation or not. Default to False.
            per_class_AR (bool): Show per class AR during evaluation or not. Default to False.
            per_attribute_per_class (bool): Show per attribute per class metrics during evaluation. Default to False.
            attribute_names (list): List of attribute names to evaluate. If None, auto-detect from annotations.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.id = 0
        self.box_id = 0
        self.id_ori = 0
        self.box_id_ori = 0
        self.gl_mode = gl_mode
        self.lframe = lframe
        self.gframe = gframe
        self.max_epoch_id = max_epoch - 1
        self.output_dir = output_dir
        self.kwargs = kwargs
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.per_attribute_per_class = per_attribute_per_class
        self.attribute_names = attribute_names
        self.vid_to_coco = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{'supercategory': 'none', 'id': i, 'name': VID_classes[i]} for i in range(len(VID_classes))],
            'images': [],
            'licenses': []
        }
        self.vid_to_coco_ori = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{'supercategory': 'none', 'id': i, 'name': VID_classes[i]} for i in range(len(VID_classes))],
            'images': [],
            'licenses': []
        }
        self.testdev = testdev
        self.tmp_name_ori = f"{self.output_dir}/ori_pred.json"
        self.tmp_name_refined = f"{self.output_dir}/refined_pred.json"
        self.gt_ori = f"{self.output_dir}/gt_ori.json"
        self.gt_refined = f"{self.output_dir}/gt_refined.json"
        self.img_id_to_name = {v:k for k,v in self.dataloader.dataset.name_id_dic.items()}

    def visualize_inferences(self, imgs, label, outputs):
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = img.cpu().detach().numpy()
            img = img.astype('uint8')  # Convert to uint8

            # Convert from (C, H, W) to (H, W, C)
            img = img.transpose(1, 2, 0)
            img = np.ascontiguousarray(img)
            # Draw bounding boxes
            if label[i] != None:
                for j in range(label[i].shape[0]):
                    cls, xmin, ymin, xmax, ymax = label[i][j]
                    xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
                    img = cv2.putText(img, str(cls.item()), (xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            if outputs[i] != None:
                for j in range(outputs[i].shape[0]):
                    xmin, ymin, xmax, ymax, obj_score, cls_score, cls = outputs[i][j]
                    xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                    score = obj_score * cls_score
                    score = round(score.item(), 3)
                    if score > 0.001:
                        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
                        # img = cv2.putText(img, str(f"{cls.item()}_{score}"), (xmin+5, ymin+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # log the image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            wandb.log({f"inferences/{i}": wandb.Image(img)})
    
    def evaluate(
            self,
            model,
            epoch=0,
            distributed=False,
            half=True,
            trt_file=None,
            decoder=None,
            test_size=None,
            img_path=None
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
        labels_list = []
        ori_data_list = []
        ori_label_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (imgs, _, info_imgs, label, path, attributes) in enumerate(
                progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                outputs, ori_res = model(imgs,
                                         lframe=self.lframe,
                                         gframe = self.gframe)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
            if self.gl_mode:
                local_num = int(imgs.shape[0] / 2)
                info_imgs = info_imgs[:local_num]
                label = label[:local_num]
            if self.kwargs.get('first_only',False):
                info_imgs = [info_imgs[0]]
                label = [label[0]]
            
            #vizualize
            if cur_iter == 0 and get_rank() == 0:
                self.visualize_inferences(imgs, label, outputs)

            temp_data_list, temp_label_list = self.convert_to_coco_format(outputs, info_imgs, copy.deepcopy(label), path, attributes)
            data_list.extend(temp_data_list) #preds
            labels_list.extend(temp_label_list) #gts

        self.vid_to_coco['annotations'].extend(labels_list)
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        # if distributed:
        #     data_list = gather(data_list, dst=0)
        #     data_list = list(itertools.chain(*data_list))
        #     torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics, epoch)
        synchronize()
        del labels_list
        del data_list
        self.vid_to_coco['annotations'] = []
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, labels, paths, attributes):
        data_list = []
        label_list = []
        frame_now = 0

        for (output, info_img, _label, path, attribute) in zip(
                outputs, info_imgs, labels, paths, attributes
        ):
            # if frame_now>=self.lframe: break
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id),
                    "image_name": path,
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3]),
                    "attributes": attribute[ind],
                }  # COCO json format
                self.box_id = self.box_id + 1
                label_list.append(label_pred_data)
            self.vid_to_coco['images'].append({'id': self.id, "file_name": path, "width": info_img[1], "height": info_img[0]})

            if output is None:
                self.id = self.id + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]
            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id),
                    "image_name": path,
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
            self.id = self.id + 1
            frame_now = frame_now + 1

        return data_list, label_list #predictions, ground_truth

    def convert_to_coco_format_ori(self, outputs, info_imgs, labels):

        data_list = []
        label_list = []
        frame_now = 0
        for (output, info_img, _label) in zip(
                outputs, info_imgs, labels
        ):
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id_ori,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3])
                }  # COCO json format
                self.box_id_ori = self.box_id_ori + 1
                label_list.append(label_pred_data)

                # print('label:',label_pred_data)

            self.vid_to_coco_ori['images'].append({'id': self.id_ori})

            if output is None:
                self.id_ori = self.id_ori + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            # print(cls.shape)
            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

            self.id_ori = self.id_ori + 1
            frame_now = frame_now + 1
        return data_list, label_list

    def evaluate_prediction(self, data_dict, statistics, epoch, ori=False):
        if not is_main_process():
            return 0, 0, None
        
        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_sampler.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_sampler.batch_size)
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

            _, tmp = tempfile.mkstemp()
            if ori:
                json.dump(self.vid_to_coco_ori, open(self.gt_ori, 'w'), indent=4)
                json.dump(data_dict, open(self.tmp_name_ori, 'w'), indent=4)
                json.dump(self.vid_to_coco_ori, open(tmp, "w"), indent=4)
            else:
                json.dump(self.vid_to_coco, open(self.gt_refined, 'w'), indent=4)
                json.dump(data_dict, open(self.tmp_name_refined, 'w'), indent=4)
                # json.dump(self.vid_to_coco, open(tmp, "w"), indent=4)

            cocoGt = pycocotools.coco.COCO(self.gt_refined)
            # TODO: since pycocotools can't process dict in py36, write data to json file.

            # _, tmp = tempfile.mkstemp()
            # json.dump(data_dict, open(tmp, "w"), indent=4)
            cocoDt = cocoGt.loadRes(self.tmp_name_refined)
            # try:
            #     from yolox.layers import COCOeval_opt as COCOeval
            # except ImportError:
            #     from pycocotools.cocoeval import COCOeval

            #     logger.warning("Use standard COCOeval.")
            from tools.cocoeval_custom import COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.params.iouThrs = np.array([0.2, 0.5, 0.75])
            cocoEval.evaluate()
            cocoEval.accumulate()
            
            redirect_string = io.StringIO()

            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            
            if self.per_class_AP:
                AP_table, per_class_AP = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
                if get_rank() == 0:
                    wandb.log({f"val/mAP_{name}":value/100 for name, value in per_class_AP.items()})

            if self.per_class_AR:
                AR_table, per_class_AR = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
                if get_rank() == 0:
                    wandb.log({f"val/mAR_{name}":value/100 for name, value in per_class_AR.items()})
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize(compute_confidence_matrix=epoch==self.max_epoch_id)
            info += redirect_string.getvalue()

            if epoch == self.max_epoch_id:
                # Log PR curve to WandB
                log_pr_curve(cocoEval, iou_threshold=0.5)
                # Log confusion matrix
                log_confusion_matrix(cocoEval, cat_names)
                
                # Per-attribute per-class evaluation
                if self.per_attribute_per_class:
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
