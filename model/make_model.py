import logging
import os
import random
from model.backbones.vit_pytorch import TransReID, deit_tiny_patch16_224_TransReID, resize_pos_embed, vit_base_patch32_224_TransReID, vit_large_patch16_224_TransReID
import torch
import torch.nn as nn

from .backbones.resnet import BasicBlock, ResNet, Bottleneck
from .backbones.resnet_ibn import resnet50_ibn_b,resnet50_ibn_a,resnet101_ibn_b,resnet101_ibn_a
from .backbones import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from .backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224


__factory_T_type = {
    'vit_large_patch16_224_TransReID': vit_large_patch16_224_TransReID,
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_base_patch32_224_TransReID': vit_base_patch32_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'deit_tiny_patch16_224_TransReID': deit_tiny_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
}

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, model_name, num_classes, cfg, num_cls_dom_wise=None):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        # model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 2048
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
            model_path = os.path.join(model_path_base, \
                "resnet18-f37072fd.pth")
            print('using resnet18 as a backbone')
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet34-b627a593.pth")
            print('using resnet34 as a backbone')
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet50-0676ba61.pth")
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
            model_path = os.path.join(model_path_base, \
                "resnet101-63fe2227.pth")
            print('using resnet101 as a backbone')
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            model_path = os.path.join(model_path_base, \
                "resnet152-394f9c45.pth")
            print('using resnet152 as a backbone')
        elif model_name == 'ibnnet50b':
            self.base = resnet50_ibn_b(pretrained=True)
        elif model_name == 'ibnnet50a':
            self.base = resnet50_ibn_a(pretrained=True)
        elif model_name == 'ibnnet101b':
            self.base = resnet101_ibn_b(pretrained=True)
        elif model_name == 'ibnnet101a':
            self.base = resnet101_ibn_a(pretrained=True)
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet' and 'ibn' not in model_name:
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # self.pool = nn.Linear(in_features=16*8, out_features=1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        #### multi-domain head
        if num_cls_dom_wise is not None:
            self.classifiers = nn.ModuleList(
                nn.Linear(self.in_planes, num_cls_dom_wise[i], bias=False)\
                    for i in range(len(num_cls_dom_wise))
            )
            for c in self.classifiers:
                c.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, domains=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x) # B, C, h, w
        
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # global_feat = self.pool(x.flatten(2)).squeeze() # is GAP harming generalization?

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            # if self.cos_layer:
            #     cls_score = self.arcface(feat, label)
            # else:
            #     cls_score = self.classifier(feat)
            # return cls_score, global_feat, label, None

            #### multi-domain head
            cls_score = self.classifier(feat)
            cls_score_ = []
            for i in range(len(self.classifiers)):
                if i not in domains:
                    cls_score_.append(None)
                    continue
                idx = torch.nonzero(domains==i).squeeze()
                cls_score_.append(self.classifiers[i](feat[idx]))
            return cls_score, global_feat, label, cls_score_
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))


class build_mix_cnn(nn.Module):
    def __init__(self, model_name, num_classes, cfg, num_cls_dom_wise):
        super(build_mix_cnn, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        # model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 2048
        if 'ibnnet' in model_name:
            if 'ibnnet50a' in model_name:
                self.base = ResNet_IBN_mix(cfg,
                                last_stride=last_stride,
                                block=Bottleneck_IBN,
                                layers=[3, 4, 6, 3],
                                ibn_cfg=('a', 'a', 'a', None))
                model_path = os.path.join(model_path_base, \
                    "resnet50_ibn_a-d9d0bb7b.pth")
                print('using mix_ibn-a-50 as a backbone')
            elif 'ibnnet50b' in model_name:
                self.base = ResNet_IBN_mix(cfg,
                                last_stride=last_stride,
                                block=Bottleneck_IBN,
                                layers=[3, 4, 6, 3],
                                ibn_cfg=('b', 'b', None, None))
                model_path = os.path.join(model_path_base, \
                    "resnet50_ibn_b-9ca61e85.pth")
                print('using mix_ibn-b-50 as a backbone')
            else:
                print("error type of ibn")
                assert False
        else:
            self.base = ResNet_mix(cfg,
                                last_stride=last_stride,
                                block=Bottleneck,
                                layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet50-0676ba61.pth")
            print('using mix_resnet50 as a backbone')

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # self.pool = nn.Linear(in_features=16*8, out_features=1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        #### multi-domain head
        if num_cls_dom_wise is not None:
            self.classifiers = nn.ModuleList(
                nn.Linear(self.in_planes, num_cls_dom_wise[i], bias=False)\
                    for i in range(len(num_cls_dom_wise))
            )
            for c in self.classifiers:
                c.apply(weights_init_classifier)

    def forward(self, x, label=None, domains=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x, domains) # B, C, h, w
        
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # global_feat = self.pool(x.flatten(2)).squeeze() # is GAP harming generalization?

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
            # #### trick from ACL
            # global_feat = nn.functional.normalize(feat,2,1)

        if self.training:
            # if self.cos_layer:
            #     cls_score = self.arcface(feat, label)
            # else:
            #     cls_score = self.classifier(feat)
            # return cls_score, global_feat, label, None
        
            #### multi-domain head
            cls_score = self.classifier(feat)
            cls_score_ = []
            for i in range(len(self.classifiers)):
                if i not in domains:
                    cls_score_.append(None)
                    continue
                idx = torch.nonzero(domains==i).squeeze()
                cls_score_.append(self.classifiers[i](feat[idx]))
            return cls_score, global_feat, label, cls_score_
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

# alter this to your pre-trained file name
lup_path_name = {
    'vit_base_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
    'vit_small_patch16_224_TransReID': 'vit_small_ics_cfs_lup.pth',
}

# alter this to your pre-trained file name
imagenet_path_name = {
    'vit_large_patch16_224_TransReID': 'jx_vit_large_p16_224-4ee7a4dc.pth',
    'vit_base_patch16_224_TransReID': 'jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_base_patch32_224_TransReID': 'jx_vit_base_patch32_224_in21k-8db57226.pth', 
    'deit_base_patch16_224_TransReID': 'deit_base_distilled_patch16_224-df68dfff.pth',
    'vit_small_patch16_224_TransReID': 'vit_small_p16_224-15ec54c9.pth',
    'deit_small_patch16_224_TransReID': 'deit_small_distilled_patch16_224-649709d9.pth',
    'deit_tiny_patch16_224_TransReID': 'deit_tiny_distilled_patch16_224-b40b3cf7.pth', 
    'swin_base_patch4_window7_224': 'swin_base_patch4_window7_224_22k.pth', 
    'swin_small_patch4_window7_224': 'swin_small_patch4_window7_224_22k.pth',
}

in_plane_dict = {
    'dhvt_tiny_patch16': 192,
    'deit_tiny_patch16_224_TransReID': 192,
    'dhvt_small_patch16': 384,
    'deit_small_patch16_224_TransReID': 384,
    'vit_small_patch16_224_TransReID': 768,
    'deit_base_patch16_224_TransReID': 768,
    'vit_base_patch16_224_TransReID': 768,
    'swin_small_patch4_window7_224': 768,
    'vit_large_patch16_224_TransReID': 1024,
    'swin_base_patch4_window7_224': 1024,
    'vit_large_patch16_224_prompt_vit': 1024,
    'vit_base_patch16_224_prompt_vit': 768,
    'vit_base_patch32_224_prompt_vit': 768,
    'deit_base_patch16_224_prompt_vit': 768,
    'vit_small_patch16_224_prompt_vit': 768,
    'deit_small_patch16_224_prompt_vit': 384,
    'deit_tiny_patch16_224_prompt_vit': 192,
}

class build_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory, num_cls_dom_wise=None):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if cfg.MODEL.TRANSFORMER_TYPE in in_plane_dict:
            self.in_planes = in_plane_dict[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            print("===== unknown transformer type =====")
            self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        if self.pretrain_choice == 'imagenet':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
                (img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
            path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
            self.model_path = os.path.join(model_path_base, path)
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
        elif self.pretrain_choice == 'LUP':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
                (img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                stem_conv=True)
            path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
            self.model_path = os.path.join(model_path_base, path)
            self.base.load_param(self.model_path)
            print('Loading pretrained LUP model......from {}'.format(self.model_path))
            
        #### original one
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        #### multi-domain head
        if num_cls_dom_wise is not None:
            self.classifiers = nn.ModuleList(
                nn.Linear(self.in_planes, num_cls_dom_wise[i])\
                    for i in range(len(num_cls_dom_wise))
            )

    def forward(self, x, domain=None):
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature

        feat = self.bottleneck(global_feat)

        if self.training:
            ### original
            cls_score = self.classifier(feat)
            return cls_score, global_feat

            # #### multi-domain head
            # cls_score = self.classifier(feat)
            # cls_score_ = []
            # for i in range(len(self.classifiers)):
            #     if i not in domain:
            #         cls_score_.append(None)
            #         continue
            #     idx = torch.nonzero(domain==i).squeeze()
            #     cls_score_.append(self.classifiers[i](feat[idx]))
            # return cls_score, global_feat, target, cls_score_

        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        count = 0
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            # if 'bottleneck' in i:
            #     continue
            if i in self.state_dict().keys():
                self.state_dict()[i].copy_(param_dict[i])
                count += 1
        print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

class build_diffusion_reid(nn.Module):
    def __init__(self, num_classes, cfg, factory, num_cls_dom_wise=None):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if cfg.MODEL.TRANSFORMER_TYPE in in_plane_dict:
            self.in_planes = in_plane_dict[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            print("===== unknown transformer type =====")
            self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        if self.pretrain_choice == 'imagenet':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
                (img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
            path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
            self.model_path = os.path.join(model_path_base, path)
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
        elif self.pretrain_choice == 'LUP':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
                (img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                stem_conv=True)
            path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
            self.model_path = os.path.join(model_path_base, path)
            self.base.load_param(self.model_path)
            print('Loading pretrained LUP model......from {}'.format(self.model_path))
            
        #### original one
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        #### multi-domain head
        if num_cls_dom_wise is not None:
            self.classifiers = nn.ModuleList(
                nn.Linear(self.in_planes, num_cls_dom_wise[i])\
                    for i in range(len(num_cls_dom_wise))
            )

    def forward(self, x, domain=None):
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature

        feat = self.bottleneck(global_feat)

        if self.training:
            ### original
            cls_score = self.classifier(feat)
            return cls_score, global_feat

            # #### multi-domain head
            # cls_score = self.classifier(feat)
            # cls_score_ = []
            # for i in range(len(self.classifiers)):
            #     if i not in domain:
            #         cls_score_.append(None)
            #         continue
            #     idx = torch.nonzero(domain==i).squeeze()
            #     cls_score_.append(self.classifiers[i](feat[idx]))
            # return cls_score, global_feat, target, cls_score_

        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        count = 0
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            # if 'bottleneck' in i:
            #     continue
            if i in self.state_dict().keys():
                self.state_dict()[i].copy_(param_dict[i])
                count += 1
        print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

def make_model(cfg, modelname, num_class, num_class_domain_wise=None):
    if modelname == 'vit':
        model = build_vit(num_class, cfg, __factory_T_type, num_class_domain_wise)
        print('===========building vit===========')
    elif modelname == 'vit':
        model = build_diffusion_reid(num_class, cfg, __factory_T_type, num_class_domain_wise)
        print('===========building vit===========')
    else:
        model = Backbone(modelname, num_class, cfg, num_class_domain_wise)
        print('===========building ResNet===========')
    ### count params
    model.compute_num_params()
    return model