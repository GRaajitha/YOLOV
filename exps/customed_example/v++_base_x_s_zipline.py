import os
import torch.nn as nn
import sys
import torch
sys.path.append("..")
from exps.yolov.yolov_base import Exp as MyExp
from loguru import logger
from yolox.data.datasets import vid
from yolox.data.data_augment import Vid_Val_Transform
from datetime import datetime

#exp after OTA_VID_woRegScore, exp 8 in the doc, decouple the reg and cls refinement
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33  # 1#0.67
        self.width = 0.5  # 1#0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.num_classes = 8  
        self.data_dir = "/shared/vision/dataset/"
        self.train_ann = "/shared/vision/dataset/metadata/v7_8_cls/coco_vid/trimmed1000_64-500seq_train_coco_vid_06_06.json"
        self.val_ann = "/shared/vision/dataset/metadata/v7_8_cls/coco_vid/trimmed1000_64-500seq_val_coco_vid_06_06.json"
        self.test_ann = "/shared/vision/dataset/metadata/v7_8_cls/coco_vid/trimmed1000_64-500seq_test_coco_vid_06_06.json"
        self.input_size = (1080, 1920)
        self.test_size = (1080, 1920)

        self.max_epoch = 20
        self.basic_lr_per_img = 0.0005 / 16
        self.warmup_epochs = 0
        self.no_aug_epochs = 2
        self.pre_no_aug = 2
        self.eval_interval = 1
        self.gmode = True
        self.lmode = False
        self.lframe = 0
        self.lframe_val = self.lframe
        self.gframe = 4
        self.gframe_val = self.gframe
        self.seq_stride = 8
        self.use_loc_emd = False
        self.iou_base = False
        self.reconf = True
        self.loc_fuse_type = 'identity'
        # self.output_dir = "./V++_outputs"
        cur_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.wandb_name = f"yolov++_base_x_s_stride{self.seq_stride}_gframe{self.gframe}_8cls_2kinp_trimmed1000_64-500seq_{cur_time}"
        self.output_dir = f"/shared/users/raajitha/YOLOVexperiments/{self.wandb_name}"
        self.stem_lr_ratio = 0.1
        self.ota_mode = True
        #check pre_nms for testing when use_pre_nms is False in training: Result: AP50 drop 3.0
        self.use_pre_nms = False
        self.cat_ota_fg = False
        self.agg_type='msa'
        self.minimal_limit = 0
        self.maximal_limit = 100
        self.decouple_reg = True
        
        # onnx_export options
        self.onnx_export=False
        self.fpn0_shape = (128, 136, 240)
        self.fpn1_shape = (256, 68, 120)
        self.fpn2_shape = (512, 34, 60)
        # topk 
        self.defualt_pre=100
        self.backbone_only = False
        self.head_only = False
        # metrics
        self.per_class_AP=True
        self.per_class_AR=True
        self.per_attribute_per_class=True
        self.attribute_names=["horizon", "size_cat", "occlusion", "clipping", "primary_terrain", "secondary_terrain", "terrain_modifier", "low_visibility", "annotated_weather", "cloud_coverage", "intruder_lateral_view", "intruder_vertical_view", "image_quality"]

    def get_model(self):
        # rewrite get model func from yolox
        if self.backbone_name == 'MCSP':
            in_channels = [256, 512, 1024]
            from yolox.models import YOLOPAFPN
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, input_size=self.input_size)
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
        from yolox.models.v_plus_head import YOLOVHead
        from yolox.models.yolov_plus import YOLOV

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
                     'iou_window':self.iou_window,'globalBlocks':self.globalBlocks,'use_pre_nms':self.use_pre_nms,
                     'cat_ota_fg':self.cat_ota_fg, 'agg_type':self.agg_type,'minimal_limit':self.minimal_limit,
                     'decouple_reg':self.decouple_reg, 'onnx_export': self.onnx_export,'maximal_limit':self.maximal_limit,
                     }
        head = YOLOVHead(self.num_classes, self.width, in_channels=in_channels, heads=self.head, drop=self.drop_rate,
                         use_score=self.use_score, defualt_p=self.defualt_p, sim_thresh=self.sim_thresh,
                         pre_nms=self.pre_nms, ave=self.ave, defulat_pre=self.defualt_pre, test_conf=self.test_conf,
                         use_mask=self.use_mask,gmode=self.gmode,lmode=self.lmode,both_mode=self.both_mode,
                         localBlocks = self.localBlocks, input_shape=self.input_size, **more_args)
        for layer in head.stems.parameters():
            layer.requires_grad = False  # set stem fixed
        for layer in head.reg_convs.parameters():
            layer.requires_grad = False
            layer.requires_grad = False
        for layer in head.cls_convs.parameters():
            layer.requires_grad = False
        for layer in head.reg_preds.parameters():
            layer.requires_grad = False

        self.model = YOLOV(backbone, head, backbone_only=self.backbone_only, head_only=self.head_only)

        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.model.apply(init_yolo)
        if self.fix_bn:
            self.model.apply(fix_bn)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2, pg3 = [], [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    if "head.stem" in k or "head.reg_convs" in k or "head.cls_convs" in k:
                        pg3.append(v.weight)
                        logger.info("head.weight: {}".format(k))
                    else:
                        pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            optimizer.add_param_group(
                {"params": pg3, "lr": lr * self.stem_lr_ratio, "weight_decay": self.weight_decay}
            )
            self.optimizer = optimizer

        return self.optimizer
    
    def get_data_loader(
            self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import TrainTransform
        dataset = vid.OVIS(   #change to your own dataset
                            img_size=self.input_size,
                            preproc=TrainTransform(
                                max_labels=50,
                                flip_prob=self.flip_prob,
                                hsv_prob=self.hsv_prob),
                            mode='uniform_w_stride',
                            lframe=self.lframe,
                            gframe=self.gframe,
                            data_dir=self.data_dir,
                            name='train',  #change to your own dir name
                            COCO_anno=os.path.join(self.data_dir, self.train_ann),
                            seq_stride=self.seq_stride)

        dataset = vid.get_trans_loader(batch_size=batch_size, data_num_workers=4, dataset=dataset)
        return dataset

    def get_eval_loader(self, batch_size,  tnum=None, data_num_workers=8, formal=False):

        assert batch_size == self.lframe_val+self.gframe_val
        dataset_val = vid.OVIS(data_dir=self.data_dir, #change to your own dataset
                               img_size=self.test_size,
                               mode='uniform_w_stride',
                               COCO_anno=os.path.join(self.data_dir, self.val_ann),
                               name='val', #change to your own dir name
                               lframe=self.lframe_val,
                               gframe=self.gframe_val,
                               preproc=Vid_Val_Transform(),
                               val=True,
                               seq_stride=self.seq_stride)

        val_loader = vid.get_trans_loader(batch_size=batch_size, data_num_workers=data_num_workers, dataset=dataset_val)
        return val_loader
