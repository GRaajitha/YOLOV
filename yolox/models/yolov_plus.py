#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn
import os
import cv2
import numpy as np
import wandb

class YOLOV(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, output_dir="./V++_Outputs"):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.output_dir = output_dir
        self.count = 0

    def forward(self, x, targets=None,nms_thresh=0.5,lframe=0,gframe=32):
        # fpn output content features of [dark3, dark4, dark5]
        if targets is not None and self.count==0:
            output_dir = f"{self.output_dir}/inputViz/"
            os.makedirs(output_dir, exist_ok=True)
            for i in range(x.shape[0]):
                img = x[i]
                img = img.cpu().detach().numpy()
                img = img.astype('uint8')  # Convert to uint8

                # Convert from (C, H, W) to (H, W, C)
                img = img.transpose(1, 2, 0)
                img = np.ascontiguousarray(img)
                # Draw bounding boxes
                for j in range(targets.shape[1]):
                    cls, c_x, c_y, w, h = targets[i, j]
                    c_x, c_y, w, h = map(int, [c_x, c_y, w/2, h/2])
                    xmin = c_x - w
                    ymin = c_y - h
                    xmax = c_x + w
                    ymax = c_y + h
                    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 3)

                # Save the image
                # cv2.imwrite(os.path.join(output_dir, f"image_{i}.png"), img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                wandb.log({f"inputs/{i}": wandb.Image(img)})
            # print(f"Saved {x.shape[0]} images in '{output_dir}' directory.")
            self.count += 1

        fpn_outs = self.backbone(x)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss,num_fg, \
            loss_refined_cls,\
            loss_refined_iou,\
            loss_refined_obj = self.head(
                fpn_outs, targets, x, lframe=lframe,gframe=gframe
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "loss_refined_cls":loss_refined_cls,
                "loss_refined_iou":loss_refined_iou,
                "loss_refined_obj":loss_refined_obj
            }
        else:
            outputs = self.head(fpn_outs,targets,x,nms_thresh=nms_thresh, lframe=lframe,gframe=gframe)

        return outputs
