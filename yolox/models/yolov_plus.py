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

    def __init__(self, backbone=None, head=None, output_dir="./V++_Outputs", backbone_only=False, head_only=False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.output_dir = output_dir
        self.count = 0
        self.backbone_only = backbone_only
        self.head_only = head_only
        assert not(self.backbone_only == True and self.head_only==True), "both backbone_only and head_only options are mutually exclusive"
        
    def forward(self, x=None, targets=None,nms_thresh=0.5,lframe=0,gframe=32, fpn_out0=None, fpn_out1=None, fpn_out2=None):
        # fpn output content features of [dark3, dark4, dark5]
        if self.training:
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
            return outputs
        else:
            if not self.backbone_only and not self.head_only:
                fpn_outs = self.backbone(x)
                outputs = self.head(fpn_outs,targets,x,nms_thresh=nms_thresh, lframe=lframe,gframe=gframe)
                return outputs
            
            if self.backbone_only:
                return self.backbone(x)

            if self.head_only:
                assert fpn_out0 is not None, "head_only option is set, must pass fpn_out0"
                assert fpn_out1 is not None, "head_only option is set, must pass fpn_out1"
                assert fpn_out2 is not None, "head_only option is set, must pass fpn_out2"
                fpn_outs = [fpn_out0, fpn_out1, fpn_out2]
                outputs = self.head(fpn_outs, targets, imgs=None, nms_thresh=nms_thresh, lframe=lframe,gframe=gframe)
                return outputs