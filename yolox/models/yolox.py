#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import os
import cv2
import numpy as np
import wandb
from yolox.utils import get_rank

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, kwargs=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(16)

        self.backbone = backbone
        self.head = head
        self.kwargs = kwargs
        self.count = 0

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        if self.count < 10 and get_rank() == 0 and targets is not None:
            # log the first image
            img = x[0]
            img = img.cpu().detach().numpy()
            
            # Undo legacy preprocessing if needed
            # Legacy preprocessing: img = img[::-1, :, :].copy() / 255.0 - mean / std
            # So we need to: (img * std + mean) * 255.0
            if self.kwargs['legacy']:
                # Reverse normalization
                mean = np.array(self.kwargs['mean']).reshape(3, 1, 1)
                std = np.array(self.kwargs['std']).reshape(3, 1, 1)
                img = img * std + mean
                img = img * 255.0
                # Reverse channel order (RGB to BGR for OpenCV)
                img = img[::-1, :, :].copy()
            
            img = img.astype('uint8')  # Convert to uint8
            # Convert from (C, H, W) to (H, W, C)
            img = img.transpose(1, 2, 0)
            img = np.ascontiguousarray(img)
            # Draw bounding boxes
            for j in range(targets[0].shape[0]):
                cls, c_x, c_y, w, h = targets[0, j]
                c_x, c_y, w, h = map(int, [c_x, c_y, w/2, h/2])
                xmin = c_x - w
                ymin = c_y - h
                xmax = c_x + w
                ymax = c_y + h
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
            # Log the image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            wandb.log({f"inputs/{self.count}": wandb.Image(img)})
            self.count += 1

        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
