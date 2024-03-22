import copy
import logging
import os
import random
from loss.metric_learning import AMSoftmax, Arcface, CircleLoss, Cosface
from model.backbones.vit_pytorch import TransReID, attr_vit_small_patch16_224_TransReID, only_attr_vit_base_patch16_224_TransReID,only_attr_vit_large_patch16_224_TransReID,attr_vit_base_patch16_224_TransReID, attr_vit_large_patch16_224_TransReID,deit_tiny_patch16_224_TransReID, resize_pos_embed, vit_base_patch32_224_TransReID, vit_large_patch16_224_TransReID
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
    "attr_vit_small_patch16_224_TransReID":attr_vit_small_patch16_224_TransReID, 
    "attr_vit_base_patch16_224_TransReID":attr_vit_base_patch16_224_TransReID, 
    "attr_vit_large_patch16_224_TransReID":attr_vit_large_patch16_224_TransReID,
    "only_attr_vit_base_patch16_224_TransReID":only_attr_vit_base_patch16_224_TransReID,
    "only_attr_vit_large_patch16_224_TransReID":only_attr_vit_large_patch16_224_TransReID,
}
factory = __factory_T_type

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

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
    def __init__(self, model_name, num_classes, cfg, num_cls_dom_wise=None, path=None):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        
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
            print('using resnet18 as a backbone')
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            print('using resnet34 as a backbone')
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
            print('using resnet101 as a backbone')
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
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
            print(model_path)
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet' and 'ibn' not in model_name:
            # print(model_path)
            # self.base = swin_base_patch4_window7_224\
            #     (img_size=cfg.INPUT.SIZE_TRAIN,
            #     stride_size=cfg.MODEL.STRIDE_SIZE,
            #     drop_path_rate=cfg.MODEL.DROP_PATH,
            #     drop_rate= cfg.MODEL.DROP_OUT,
            #     attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
            if path:
                self.load_param(path)
                print('Loading trained model......from {}'.format(path))
            else:
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
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
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
    'attr_vit_large_patch16_224_TransReID':1024,
    'attr_vit_small_patch16_224_TransReID':768,
}

class build_vit(nn.Module):
    def __init__(self, num_classes, cfg, model_name=None, num_cls_dom_wise=None, pretrain_choice='imagenet', model_path=None):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        self.pretrain_choice = pretrain_choice
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if not model_name:
            model_name = cfg.MODEL.TRANSFORMER_TYPE
        if model_name in in_plane_dict:
            self.in_planes = in_plane_dict[model_name]
        else:
            print("===== unknown transformer type =====")
            self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        if self.pretrain_choice == 'imagenet':
            self.base = __factory_T_type[model_name]\
                (img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        elif self.pretrain_choice == 'LUP':
            self.base = __factory_T_type[model_name]\
                (img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                stem_conv=True)
        if model_path:
            self.model_path = model_path
            self.load_param(model_path)
        else:
            self.model_path = model_path_base
            self.base.load_param(self.model_path)
        print('Loading pretrained model......from {}'.format(self.model_path))
            # path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
            # self.model_path = os.path.join(model_path_base, path)
            # self.base.load_param(self.model_path)
            # print('Loading pretrained LUP model......from {}'.format(self.model_path))
            
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

class build_attr_vit(nn.Module):
    def __init__(self, num_classes, cfg, model_name=None, pretrain_choice='imagenet', stride_size=16, model_path=None, img_size=None):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        self.pretrain_choice = pretrain_choice
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if not model_name:
            model_name = cfg.MODEL.TRANSFORMER_TYPE
        if model_name in in_plane_dict:
            self.in_planes = in_plane_dict[model_name]
        else:
            print("===== unknown transformer type =====")
            self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        img_size = cfg.INPUT.SIZE_TRAIN if not img_size else img_size
        if self.pretrain_choice in ['imagenet', 'self']:
            self.base = factory[model_name]\
                (img_size=img_size,
                stride_size=stride_size,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                has_attr_emb=cfg.MODEL.HAS_ATTRIBUTE_EMBEDDING)
        elif self.pretrain_choice == 'LUP':
            self.base = factory[model_name]\
                (img_size=img_size,
                stride_size=stride_size,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                stem_conv=True,
                has_attr_emb=cfg.MODEL.HAS_ATTRIBUTE_EMBEDDING)
        # import ipdb; ipdb.set_trace()
        if model_path:
            self.model_path = model_path
            self.load_param(model_path)
        elif pretrain_choice == 'self':
            self.model_path = model_path_base
            self.load_param(model_path_base)
        else:
            self.model_path = model_path_base
            self.base.load_param(self.model_path)
        print('Loading pretrained model......from {}'.format(self.model_path))
        
        #### original one
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.attr_head = nn.ModuleList([
            nn.Linear(self.in_planes, 2, bias=False), # gender
            nn.Linear(self.in_planes, 5, bias=False), # backpack
            nn.Linear(self.in_planes, 5, bias=False), # hat
            nn.Linear(self.in_planes, 12, bias=False), # upper cloth color 
            nn.Linear(self.in_planes, 4, bias=False), # upper cloth style 
            nn.Linear(self.in_planes, 12, bias=False), # lower cloth color 
            nn.Linear(self.in_planes, 4, bias=False), # lower cloth style 
        ])
        for h in self.attr_head:
            h.apply(weights_init_classifier)
            

    def forward(self, x, attr_recognition=False, attrs=None):
        x = self.base(x, attrs) # B, N, C
        global_feat = x[:, 0] # cls token for global feature
        attr_tokens = x[:, 1:8]

        feat = self.bottleneck(global_feat)

        if attr_recognition or self.training:
            attr_scores = []
            for i in range(7):
                score = self.attr_head[i](attr_tokens[:, i])
                attr_scores.append(score)

        # import ipdb; ipdb.set_trace()

        if self.training:
            ### original
            cls_score = self.classifier(feat)
            return cls_score, global_feat, attr_scores
        else:
            if attr_recognition:
                return x, attr_scores
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
                if 'pos_embed' in i and param_dict[i].shape != self.base.pos_embed.shape:
                    param_dict[i] = self.base.resize_pos_embed(param_dict[i], self.base.pos_embed, self.base.patch_embed.num_y, self.base.patch_embed.num_x, [21, 10] if self.pretrain_choice=='self' else None)
                self.state_dict()[i].copy_(param_dict[i])
                count += 1
        print('Loading trained model from {}\n Load {}/{} layers'.format(trained_path, count, len(self.state_dict())))
        
    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))
        

## only cls token for multi-task classification
class build_attr_vit_V2(nn.Module):
    def __init__(self, num_classes, cfg, factory):
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
        elif self.pretrain_choice == 'LUP':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
                (img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                stem_conv=True)
        self.model_path = model_path_base
        self.base.load_param(self.model_path)
        print('Loading pretrained model......from {}'.format(self.model_path))
            
        #### original one
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.attr_head = nn.ModuleList([
            nn.Linear(self.in_planes, 2, bias=False), # gender
            nn.Linear(self.in_planes, 5, bias=False), # backpack
            nn.Linear(self.in_planes, 5, bias=False), # hat
            nn.Linear(self.in_planes, 12, bias=False), # upper cloth color 
            nn.Linear(self.in_planes, 4, bias=False), # upper cloth style 
            nn.Linear(self.in_planes, 12, bias=False), # lower cloth color 
            nn.Linear(self.in_planes, 4, bias=False), # lower cloth style 
        ])
        for h in self.attr_head:
            h.apply(weights_init_classifier)

    def forward(self, x, attr_recognition=False):
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature

        feat = self.bottleneck(global_feat)

        attr_scores = []
        for i in range(7):
            score = self.attr_head[i](global_feat)
            attr_scores.append(score)

        # import ipdb; ipdb.set_trace() 

        if self.training:
            ### original
            cls_score = self.classifier(feat)
            return cls_score, global_feat, attr_scores
        else:
            if attr_recognition:
                return x[:, :8], attr_scores
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
        
    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

class build_transformer_local(nn.Module):
    def __init__(self, num_classes, cfg, factory, attr_tokens=False):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.attr_tokens = attr_tokens

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, local_feature=cfg.MODEL.JPM, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = cfg.MODEL.RE_ARRANGE

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        features = self.base(x)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                # return torch.cat(
                #     [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

class build_only_attr_vit_cls(nn.Module):
    def __init__(self, num_classes, cfg, factory):
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
        # self.num_classes = num_classes

        if self.pretrain_choice == 'imagenet':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
                (img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        elif self.pretrain_choice == 'LUP':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
                (img_size=cfg.INPUT.SIZE_TRAIN,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate= cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                stem_conv=True)
        self.model_path = model_path_base
        self.base.load_param(self.model_path)
        print('Loading pretrained model......from {}'.format(self.model_path))
            
        #### original one
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # self.classifier.apply(weights_init_classifier)
        # self.bottleneck = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck.bias.requires_grad_(False)
        # self.bottleneck.apply(weights_init_kaiming)

        self.attr_head = nn.ModuleList([
            nn.Linear(self.in_planes, 2, bias=False), # gender
            nn.Linear(self.in_planes, 5, bias=False), # backpack
            nn.Linear(self.in_planes, 5, bias=False), # hat
            nn.Linear(self.in_planes, 12, bias=False), # upper cloth color 
            nn.Linear(self.in_planes, 4, bias=False), # upper cloth style 
            nn.Linear(self.in_planes, 12, bias=False), # lower cloth color 
            nn.Linear(self.in_planes, 4, bias=False), # lower cloth style 
        ])
        for h in self.attr_head:
            h.apply(weights_init_classifier)

    def forward(self, x, attr_recognition=False):
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature
        attr_tokens = x[:, 1:8]

        # feat = self.bottleneck(global_feat)

        attr_scores = []
        for i in range(7):
            score = self.attr_head[i](attr_tokens[:, i])
            attr_scores.append(score)

        # import ipdb; ipdb.set_trace() 

        if self.training:
            ### original
            # cls_score = self.classifier(feat)
            # return cls_score, global_feat, attr_scores
            return attr_scores
        else:
            if attr_recognition:
                return x, attr_scores
            # return feat if self.neck_feat == 'after' else global_feat

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
        
    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('reid.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))
 

def make_model(cfg, modelname, num_class, num_class_domain_wise=None):
    pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
    if modelname == 'vit':
        model = build_vit(num_class, cfg, None, num_class_domain_wise, pretrain_choice)
        print('===========building vit===========')
    elif modelname == 'local_vit':
        model = build_transformer_local(num_class,cfg,__factory_T_type)
        print('===========building vit with JPM===========')
    elif modelname == 'attr_vit':
        model = build_attr_vit(num_class, cfg, None, pretrain_choice, cfg.MODEL.STRIDE_SIZE)
        print('===========building attr_vit===========')
    elif modelname == 'attr_vit_only_cls':
        model = build_attr_vit_V2(num_class, cfg, __factory_T_type)
        print('===========building attr_vit===========')
    elif modelname == "only_attribute_recognition":
        model = build_only_attr_vit_cls(num_class, cfg, __factory_T_type)
        print('===========building only_attribute_recognition===========')
    else:
        model = Backbone(modelname, num_class, cfg, num_class_domain_wise)
        print('===========building ResNet===========')
    ### count params
    model.compute_num_params()
    return model