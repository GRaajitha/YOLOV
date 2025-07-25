import os
from datetime import datetime
from yolox.exp import Exp as MyExp
import torch
import torch.nn as nn
from datetime import date

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.num_classes = 8  
        self.data_dir = "/shared/vision/dataset/"
        self.train_ann = "metadata/v7_8_cls/coco_vid/trimmed1000_64-500seq_train_coco_vid_06_06.json"
        self.val_ann = "metadata/v7_8_cls/coco_vid/trimmed1000_64-500seq_val_coco_vid_06_06.json"
        self.test_ann = "metadata/v7_8_cls/coco_vid/trimmed1000_64-500seq_test_coco_vid_06_06.json"
        self.input_size = (1080, 1920)
        self.test_size = (1080, 1920)
        self.train_name = ''
        self.val_name = ''
        self.wandb_name = f"yoloxs_v7_8cls_1080x1920_20ep_trimmed1000_64-500seq_{date.today()}"
        self.output_dir = f"/shared/users/raajitha/YOLOVexperiments/{self.wandb_name}"

        self.max_epoch = 20
        self.no_aug_epochs = 10
        self.warmup_epochs = 3
        self.eval_interval = 1
        self.print_interval = 10
        self.min_lr_ratio = 0.00000005
        self.basic_lr_per_img = 0.00003125
        self.multiscale_range = 5
        self.test_conf = 0.001
        self.nmsthre = 0.5
        self.data_num_workers = 6
        self.momentum = 0.9
        #COCO API has been changed
        self.data_shuffle = False
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        # metrics
        self.per_class_AP=True
        self.per_class_AR=True
        self.per_attribute_per_class=True
        self.attribute_names=["horizon", "size_cat", "occlusion", "clipping", "primary_terrain", "secondary_terrain", "terrain_modifier", "low_visibility", "annotated_weather", "cloud_coverage", "intruder_lateral_view", "intruder_vertical_view", "image_quality"]
        #ONNX export
        self.onnx_export = False
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size
            pg0, pg1, pg2,pg3 = [], [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v,'absolute_pos_embed') or hasattr(v,'relative_position_bias_table') or hasattr(v,'norm'):
                    if hasattr(v,'weight'):
                        pg3.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.AdamW(params=pg0,lr=lr,weight_decay=self.weight_decay)
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group(
                {"params": pg3, "weight_decay": 0}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})

            self.optimizer = optimizer

        return self.optimizer