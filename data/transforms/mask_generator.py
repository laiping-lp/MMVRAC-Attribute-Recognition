"""
stole from MAE-pytorch
https://github.com/pengzhiliang/MAE-pytorch
"""
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np
import torch

# for Part_Attention, add by me
def generate_2d_mask(H=16, W=8, left=0, top=0, width=8, height=8, part=-1):
    H, W, left, top, width, height = \
        int(H), int(W), int(left), int(top), int(width), int(height)
    assert left + width <= W and top + height <= H

    l,w,t,h = left, left+width, top, top+height ### for test
    mask = np.zeros([H, W])
    mask[t : h + 1, l : w + 1] = 1
    mask = mask.flatten(order='C')
    mask_ = np.zeros([len(mask) + 4])
    mask_[4:] = mask
    mask_[part] = 1
    mask_[0] = 1 ######### cls token
    mask_ = mask_[:, np.newaxis] # N x 1
    mask_ = mask_ @ mask_.transpose() # N x N
    return mask_.astype(bool)

class RandomMaskingGenerator:
    def __init__(self, cfg, mask_ratio):
        self.height, self.width = cfg.INPUT.SIZE_TRAIN
        self.stride = cfg.MODEL.STRIDE_SIZE
        self.num_patches = (self.height // self.stride) * (self.width // self.stride)
        self.num_mask = int(mask_ratio * self.num_patches)
        self.mask_ratio = mask_ratio

    def __repr__(self):
        repr_str = "Mask ratio: {}, total patches {}, mask patches {}".format(
            self.mask_ratio, self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        mask = np.hstack([np.zeros(1), mask]) ###### add by me
        return mask # [196]
    
class PartAwareMaskGenerator: ##### to do
    def __init__(self, cfg, part_num=3):
        self.height, self.width = cfg.INPUT.SIZE_TRAIN
        self.stride = cfg.MODEL.STRIDE_SIZE
        self.num_patches = (self.height // self.stride) * (self.width // self.stride) + 1 + part_num
        self.part_num = part_num

    def __call__(self):
        n = self.num_patches
        k = self.part_num
        H, W = self.height // self.stride, self.width //self.stride
        mask = np.ones(shape=(n,n))
        mask[1 : k+1, 0] = 0
        mask_ = (mask @ mask.transpose()).astype(bool)
        for i in range(k):
            mask_ |= generate_2d_mask(H,W,0,i*H/k,W,H/k,i+1)
        mask_[1 : k+1, 0] = True
        mask_[0, 1 : k+1] = True
        return mask_[np.newaxis,:,:] # 1 x patch num x patch num