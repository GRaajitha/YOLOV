import os
import torch
import torch.nn as nn

from yolox.exp import Exp as MyExp
from datetime import date

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Define yourself dataset path
        self.num_classes = 8  
        self.data_dir = "/shared/vision/dataset/"
        self.train_ann = "metadata/v7_8_cls/train_annotations_coco_fmt.json"
        self.val_ann = "metadata/v7_8_cls/val_annotations_coco_fmt.json"
        self.test_ann = "metadata/v7_8_cls/test_annotations_coco_fmt.json"
        self.input_size = (1080, 1920)
        self.test_size = (1080, 1920)
        self.train_name = ''
        self.val_name = ''
        self.wandb_name = f"yoloxs_v7_8cls_1080x1920_20ep_{date.today()}"
        self.output_dir = f"/shared/users/raajitha/YOLOVexperiments/{self.wandb_name}"

        self.basic_lr_per_img = 0.0005 / 32.0
        self.save_history_ckpt = False
        self.max_epoch = 20
        self.eval_interval = 1
        self.warmup_epochs = 3
        self.no_aug_epochs = 10
        self.test_conf = 0.001
        self.nmsthre = 0.5
        self.data_num_workers = 6
        self.momentum = 0.9
        self.mosaic_scale = (0.1, 2)
        self.enable_mixup = False

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [192, 384, 768]
        out_channels = [192, 384, 768]
        from yolox.models import YOLOX, YOLOPAFPN_Swin, YOLOXHead
        backbone = YOLOPAFPN_Swin(in_channels=in_channels, out_channels=out_channels, act=self.act,in_features=(1,2,3), input_size=self.input_size)
        head = YOLOXHead(self.num_classes, self.width, in_channels=out_channels, act=self.act)
        self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model

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

    #
    # def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
    #     from yolox.evaluators import COCOEvaluator
    #
    #     val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
    #     evaluator = COCOEvaluator(
    #         dataloader=val_loader,
    #         img_size=self.test_size,
    #         confthre=self.test_conf,
    #         nmsthre=self.nmsthre,
    #         num_classes=self.num_classes,
    #         testdev=testdev,
    #         fg_AR_only=True,
    #     )
    #     return evaluator