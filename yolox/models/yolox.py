#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import os
import cv2
import numpy as np

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(16)

        self.backbone = backbone
        self.head = head
        self.count = 1

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        if targets is not None and self.count==0:
            output_dir = f"{self.output_dir}/yolox_inputViz/"
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
