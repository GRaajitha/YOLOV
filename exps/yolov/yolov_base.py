#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.data.datasets import vid
from yolox.exp.base_exp import BaseExp
from yolox.data.data_augment import Vid_Val_Transform


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.archi_name = 'YOLOV'
        self.backbone_name = 'MCSP'
        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 30
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"
        # ---------------- swintrans backbone config --------- #
        self.pretrain_img_size = 224
        self.window_size = 7
        # ---------------- focalnet backbone config --------- #
        self.focal_level = 4
        self.focal_windows = 3
        self.focal_fpn_channels = [256, 512, 1024]

        # ---------------- yolov config ---------------- #
        # drop out rate for multi head attention
        self.drop_rate = 0
        # multi head number
        self.head = 4
        # defualt proposal number per frame
        self.defualt_p = 30
        # similarity thresh hold for ave pooling
        self.sim_thresh = 0.75
        # first stage preposal nms threshold
        self.pre_nms = 0.75
        # use ave pooling
        self.ave = True
        # topK proposal number for first stage
        self.defualt_pre = 750
        # use confidence score or not
        self.use_score = True
        # old version legacy
        self.perspective = 0.0
        # fix backbone bn
        self.fix_bn = False
        # use strong augmentation or not
        self.use_aug = False
        # use confidence mask or not
        self.use_mask = False
        # fix all vallina param
        self.fix_all = False
        # global feature fusion
        self.gmode = True
        #local feature fusion
        self.lmode = False
        # both local and global feature fusion
        self.both_mode = False
        #lframe for training
        self.lframe = 0
        #lframe for validation
        self.lframe_val = 0
        #local block number
        self.localBlocks = 1
        #global frames for training
        self.gframe = 16
        #globale frames for validation
        self.gframe_val = 32
        #sequence number for validation,-1 denote all
        self.tnum = -1
        #
        self.local_stride = 1
        #
        self.iou_window = 0
        #
        self.globalBlocks = 1

        # ---------------- local mode config --------- #
        #ffn in local blocks
        self.use_ffn = True
        #time emb in local blocks
        self.use_time_emd = False
        # loc emb in local blocks, ref to RelationDet
        self.use_loc_emd = True
        # fuse type for QK and local sim
        self.loc_fuse_type = 'add'
        #qkv or only linear
        self.use_qkv = True
        # mask reference features in local frames
        self.local_mask = False
        #which branch to mask for local_mask, cls or reg or ''
        self.local_mask_branch = ''
        #convert the xyxy to pure pos emb if true
        self.pure_pos_emb = False
        #use conf guided feature agg in local blocks or not
        self.loc_conf = False
        #use iou based feature agg in local blocks or not, if ture will set use_loc_emb,use_time_emb and pure_pos_emb to False
        self.iou_base = False
        # try to re-confidence the iou score in reg branch
        self.reconf = False
        #only use ota assignment lables for supervision in re-conf
        self.ota_mode = False
        #only use ota assignment lables for supervision in re-cls
        self.ota_cls = False
        #traj linking for post-processing before the second nms
        self.traj_linking = False
        #minimal limitation for candidates
        self.minimal_limit = 0
        #add vid cls branch
        self.vid_cls = True
        #add vid reg branch
        self.vid_reg = False
        #threshold for reg agg
        self.conf_sim_thresh = 0.99

        # ---------------- dataloader config ---------------- #
        # set worker to 12 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 12
        self.input_size = (576, 576)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = '/mnt/weka/scratch/yuheng.shi/dataset/VID'
        # name of annotation file for training
        self.vid_train_path = ''
        self.vid_val_path = ''
        # path to vid name list

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #

        # epoch number used for warmup
        self.warmup_epochs = 1
        # max training epoch
        self.max_epoch = 7
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.1
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.002 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 2
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (1920, 1920)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.001
        # nms threshold
        self.nmsthre = 0.5
        
        # ---------------- evaluation config ------------------ #
        # per class AP and AR metrics
        self.per_class_AP = False
        self.per_class_AR = False
        # per attribute per class metrics
        self.per_attribute_per_class = False
        self.attribute_names = None  # List of attribute names to evaluate. If None, auto-detect from annotations.

    def get_model(self):
        # rewrite get model func from yolox
        if self.backbone_name == 'MCSP':
            in_channels = [256, 512, 1024]
            from yolox.models import YOLOPAFPN
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
        elif 'Swin' in self.backbone_name:
            from yolox.models import YOLOPAFPN_Swin

            if self.backbone_name == 'Swin_Tiny':
                in_channels = [192, 384, 768]
                out_channels = [192, 384, 768]
                backbone = YOLOPAFPN_Swin(in_channels=in_channels,
                                          out_channels=out_channels,
                                          act=self.act,
                                          in_features=(1, 2, 3))
            elif self.backbone_name == 'Swin_Base':
                in_channels = [256, 512, 1024]
                out_channels = [256, 512, 1024]
                backbone = YOLOPAFPN_Swin(in_channels=in_channels,
                                          out_channels=out_channels,
                                          act=self.act,
                                          in_features=(1, 2, 3),
                                          swin_depth=[2, 2, 18, 2],
                                          num_heads=[4, 8, 16, 32],
                                          base_dim=int(in_channels[0] / 2),
                                          pretrain_img_size=self.pretrain_img_size,
                                          window_size=self.window_size,
                                          width=self.width,
                                          depth=self.depth
                                          )
        elif 'Focal' in self.backbone_name:
            from yolox.models import YOLOPAFPN_focal
            fpn_in_channles = [96 * 4, 96 * 8, 96 * 16]
            in_channels = self.focal_fpn_channels
            backbone = YOLOPAFPN_focal(in_channels=fpn_in_channles,
                                       out_channels=in_channels,
                                       act=self.act,
                                       in_features=(1, 2, 3),
                                       depths=[2, 2, 18, 2],
                                       focal_levels=[4, 4, 4, 4],
                                       focal_windows=[3, 3, 3, 3],
                                       use_conv_embed=True,
                                       use_postln=True,
                                       use_postln_in_modulation=False,
                                       use_layerscale=True,
                                       base_dim=192,  # int(in_channels[0])
                                       depth=self.depth,
                                       width=self.width
                                       )


        else:
            raise NotImplementedError('backbone not support')
        from yolox.models.yolovp_msa import YOLOXHead
        from yolox.models.myolox import YOLOX

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03


        for layer in backbone.parameters():
            layer.requires_grad = False  # fix the backbone
        more_args = {'use_ffn': self.use_ffn, 'use_time_emd': self.use_time_emd, 'use_loc_emd': self.use_loc_emd,
                     'loc_fuse_type': self.loc_fuse_type, 'use_qkv': self.use_qkv,
                     'local_mask': self.local_mask, 'local_mask_branch': self.local_mask_branch,
                     'pure_pos_emb':self.pure_pos_emb,'loc_conf':self.loc_conf,'iou_base':self.iou_base,
                     'reconf':self.reconf,'ota_mode':self.ota_mode,'ota_cls':self.ota_cls,'traj_linking':self.traj_linking,
                     'iou_window':self.iou_window,'globalBlocks':self.globalBlocks,'minimal_limit':self.minimal_limit,
                     'vid_cls':self.vid_cls,'vid_reg':self.vid_reg,'conf_sim_thresh':self.conf_sim_thresh,
                     }
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, heads=self.head, drop=self.drop_rate,
                         use_score=self.use_score, defualt_p=self.defualt_p, sim_thresh=self.sim_thresh,
                         pre_nms=self.pre_nms, ave=self.ave, defulat_pre=self.defualt_pre, test_conf=self.test_conf,
                         use_mask=self.use_mask,gmode=self.gmode,lmode=self.lmode,both_mode=self.both_mode,
                         localBlocks = self.localBlocks,**more_args)

        for layer in head.stems.parameters():
            layer.requires_grad = False  # set stem fixed
        for layer in head.reg_convs.parameters():
            layer.requires_grad = False
            layer.requires_grad = False
        for layer in head.cls_convs.parameters():
            layer.requires_grad = False
        for layer in head.reg_preds.parameters():
            layer.requires_grad = False
        if self.fix_all:
            for layer in head.obj_preds.parameters():
                layer.requires_grad = False
            for layer in head.cls_preds.parameters():
                layer.requires_grad = False
        self.model = YOLOX(backbone, head)

        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.model.apply(init_yolo)
        if self.fix_bn:
            self.model.apply(fix_bn)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
            self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import TrainTransform
        from yolox.data.datasets.mosaicdetection import MosaicDetection_VID
        assert batch_size == self.lframe + self.gframe
        dataset = vid.VIDDataset(file_path=self.vid_train_path,
                                 img_size=self.input_size,
                                 preproc=TrainTransform(
                                     max_labels=50,
                                     flip_prob=self.flip_prob,
                                     hsv_prob=self.hsv_prob),
                                 lframe=self.lframe,  # batch_size,
                                 gframe=self.gframe,
                                 dataset_pth=self.data_dir,
                                 local_stride=self.local_stride,
                                 )
        if self.use_aug:
            # NO strong aug by defualt
            dataset = MosaicDetection_VID(
                dataset,
                mosaic=False,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=120,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,
                perspective=self.perspective,
                enable_mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
                dataset_path=self.data_dir
            )
        dataset = vid.get_trans_loader(batch_size=batch_size, data_num_workers=4, dataset=dataset)
        return dataset

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, tnum=None, data_num_workers=8,formal=False):
        if tnum == None:
            tnum = self.tnum
        assert batch_size == self.lframe_val+self.gframe_val
        dataset_val = vid.VIDDataset(file_path=self.vid_val_path,
                                     img_size=self.test_size, preproc=Vid_Val_Transform(), lframe=self.lframe_val,
                                     gframe=self.gframe_val, val=True, dataset_pth=self.data_dir, tnum=tnum,formal=formal,
                                     traj_linking=self.traj_linking, local_stride=self.local_stride,)
        val_loader = vid.vid_val_loader(batch_size=batch_size,
                                        data_num_workers=data_num_workers,
                                        dataset=dataset_val, )

        return val_loader

    # rewrite evaluation func
    def get_evaluator(self, val_loader):
        from yolox.evaluators.vid_evaluator_v2 import VIDEvaluator
        # val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VIDEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            lframe=self.lframe_val,
            gframe=self.gframe_val,
            first_only = False,
            max_epoch=self.max_epoch,
            output_dir=self.output_dir,
            per_class_AP=self.per_class_AP,
            per_class_AR=self.per_class_AR,
            per_attribute_per_class=self.per_attribute_per_class,
            attribute_names=self.attribute_names,
        )
        return evaluator

    def get_trainer(self, args):
        from yolox.core import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, epoch, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, epoch, is_distributed, half)
