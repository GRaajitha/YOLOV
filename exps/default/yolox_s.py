#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (1920, 1920)
        self.test_size = (1920, 1920)
        self.multiscale_range = 0

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/shared/vision/dataset/"
        self.train_ann = "metadata/v7/subsample_10_percent/train_annotations_coco_fmt.json"
        self.val_ann = "metadata/v7/subsample_10_percent/val_annotations_coco_fmt.json"

        self.num_classes = 16

        self.max_epoch = 300
        self.eval_interval = 1
        self.warmup_epochs = 1
        self.no_aug_epochs = 7
        self.train_name = ''
        self.val_name = ''
