import os
from datetime import datetime
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path

        self.num_classes = 16
        self.data_dir = "/shared/vision/dataset/"
        self.train_ann = "metadata/v7/subsample_10_percent/train_annotations_coco_fmt.json"
        self.val_ann = "metadata/v7/subsample_10_percent/val_annotations_coco_fmt.json"
        self.test_ann = "metadata/v7/subsample_10_percent/test_annotations_coco_fmt.json"
        self.output_dir = "/shared/users/raajitha/YOLOVexperiments/yoloxs_vid_2kinp_0_2nms_jan21"

        self.train_name = ''
        self.val_name = ''
        self.max_epoch = 30
        self.no_aug_epochs = 1
        self.warmup_epochs = 0
        self.eval_interval = 1
        self.print_interval = 10
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.0001
        self.input_size = (1920, 1920)
        self.test_size = (1920, 1920)
        self.multiscale_range = 0
        self.test_conf = 0.001
        self.nmsthre = 0.2
        self.data_num_workers = 6
        # self.momentum = 0.9
        #COCO API has been changed

    
