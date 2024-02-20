"""
    stole from Liao: Graph Sampling Based Deep Metric Learning for
    Generalizable Person Re-Identification, CVPR2022
"""

from __future__ import absolute_import
from collections import OrderedDict, defaultdict
import time
from random import shuffle
from tracemalloc import is_tracing
# from cv2 import transform
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from data.datasets.bases import BaseDataset, Dataset, ImageDataset

# from .preprocessing import Preprocessor
# from reid.evaluators import extract_features, pairwise_distance

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat

def extract_batch_feature(model, inputs):
    with torch.no_grad():
        outputs = model(inputs.cuda())

    # f = outputs[1]
    # outputs = outputs[0]
    outputs = outputs.cpu()
    # return outputs, f
    return outputs


def extract_features(model, data_loader, verbose=False):
    fea_time = 0
    data_time = 0
    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()

    if verbose:
        print('Extract Features...', end='\t')

    if 'patch_embed' in dict(model.base.named_children()).keys():
        is_train = model.base.patch_embed.training
    model = model.cuda().eval()
    with torch.no_grad():
        for i, (imgs, pids, camids, others, fnames) in enumerate(data_loader):
            data_time += time.time() - end
            end = time.time()

            # outputs, f = extract_cnn_feature(model, imgs)
            outputs = extract_batch_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            fea_time += time.time() - end
            end = time.time()
    model = model.train()
    if 'patch_embed' in dict(model.base.named_children()).keys():
        model.base.patch_embed.train(is_train)

    if verbose:
        print('Feature time: {:.3f} seconds. Data time: {:.3f} seconds.'.format(fea_time, data_time))

    # return features, labels, f
    return features, labels

class GraphSampler(Sampler):
    def __init__(self, data_source, model, batch_size=64, num_instance=4, num_workers=8, transform=None, 
                gal_batch_size=256, prob_batch_size=256, save_path=None, verbose=True):
        super(GraphSampler, self).__init__(data_source)
        self.data_source = data_source
        # self.img_path = img_path
        # self.transformer = transformer
        self.model = model
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.num_workers = num_workers
        self.gal_batch_size = gal_batch_size
        self.prob_batch_size = prob_batch_size
        self.save_path = save_path
        self.verbose = verbose
        self.transform = transform
        self.epoch = 0

        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_pids = len(self.pids)
        for pid in self.pids:
            shuffle(self.index_dic[pid])

        self.sam_index = None
        self.sam_pointer = [0] * self.num_pids

    def make_index(self):
        start = time.time()
        self.graph_index()
        if self.verbose:
            print('\nTotal GS time: %.3f seconds.\n' % (time.time() - start))

    def calc_distance(self, dataset):
        data_loader = DataLoader(
            dataset=BaseDataset(dataset,[],[], self.transform),
            batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=True)

        if self.verbose:
            print('\t GraphSampler: ', end='\t')
        features, _ = extract_features(self.model, data_loader, self.verbose)
        # data_num x 24h x 8w x 1536d
        features = torch.cat([features[fname.split('/')[-1]].unsqueeze(0) for fname, _, _, _ in dataset], 0)

        if self.verbose:
            print('\t GraphSampler: \tCompute distance...', end='\t')
        start = time.time()
        # dist_compute
        dist = euclidean_distance(features, features)
        
        if self.verbose:
            print('Time: %.3f seconds.' % (time.time() - start))

        return dist

    def graph_index(self):
        sam_index = []
        for pid in self.pids:
            # 随机选取这个人的一张图片
            index = np.random.choice(self.index_dic[pid], size=1)[0]
            sam_index.append(index)

        # 每人一张图片
        dataset = [self.data_source[i] for i in sam_index]
        dist = self.calc_distance(dataset)
        with torch.no_grad():
            # 对角线加上1e15，同一ID距离无限大
            dist = dist + torch.eye(self.num_pids, device=dist.device) * 1e15
            topk = self.batch_size // self.num_instance - 1
            # 返回前topk个距离最小的
            _, topk_index = torch.topk(dist.cuda(), topk, largest=False)
            topk_index = topk_index.cpu().numpy()            

        if self.save_path is not None:
            filenames = [fname for fname, _, _, _ in dataset]
            test_file = os.path.join(self.save_path, 'gs%d.npz' % self.epoch)
            np.savez_compressed(test_file, filenames=filenames, dist=dist.cpu().numpy(), topk_index=topk_index)

        sam_index = []
        for i in range(self.num_pids):
            id_index = topk_index[i, :].tolist()
            id_index.append(i)
            index = []
            for j in id_index:
                pid = self.pids[j]
                img_index = self.index_dic[pid]
                len_p = len(img_index)
                index_p = []
                remain = self.num_instance
                while remain > 0:
                    end = self.sam_pointer[j] + remain
                    idx = img_index[self.sam_pointer[j] : end]
                    index_p.extend(idx)
                    remain -= len(idx)
                    self.sam_pointer[j] = end
                    if end >= len_p:
                        shuffle(img_index)
                        self.sam_pointer[j] = 0
                assert(len(index_p) == self.num_instance)
                index.extend(index_p)
            sam_index.extend(index)

        sam_index = np.array(sam_index)
        sam_index = sam_index.reshape((-1, self.batch_size))
        np.random.shuffle(sam_index)
        sam_index = list(sam_index.flatten())
        self.sam_index = sam_index

    def __len__(self):
        if self.sam_index is None:
            return self.num_pids
        else:
            return len(self.sam_index)

    def __iter__(self):
        self.make_index()
        # print(1)
        self.epoch = self.epoch + 1
        return iter(self.sam_index)
