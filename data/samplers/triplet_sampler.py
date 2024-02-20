import copy
import itertools
from collections import defaultdict
import logging
import random
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from data.samplers.graph_sampler import GraphSampler
from loss.triplet_loss import euclidean_dist

from utils import comm
import os

def no_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class BalancedIdentitySampler(Sampler):
    '''
    for each id, don't select images from the same camera/domain
    '''
    def __init__(self, data_source: str, batch_size: int, num_instances: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            # camid = info[2]
            camid = info[3]['domains']
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def _get_epoch_indices(self):
        # Shuffle identity list
        identities = np.random.permutation(self.num_identities)

        # If remaining identities cannot be enough for a batch,
        # just drop the remaining parts
        drop_indices = self.num_identities % self.num_pids_per_batch
        if drop_indices: identities = identities[:-drop_indices]

        ret = []
        for kid in identities:
            i = np.random.choice(self.pid_index[self.pids[kid]])
            i_cam = self.data_source[i][3]['domains']
            # _, i_pid, i_cam = self.data_source[i]
            ret.append(i)
            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = no_index(cams, i_cam)

            if select_cams:
                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                for kk in cam_indexes:
                    ret.append(index[kk])
            else:
                select_indexes = no_index(index, i)
                if not select_indexes:
                    # only one image for this identity
                    ind_indexes = [0] * (self.num_instances - 1)
                elif len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return ret

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            indices = self._get_epoch_indices()
            yield from indices


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source: str, batch_size: int, num_instances: int, delete_rem: bool, seed: Optional[int] = None, cfg = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.delete_rem = delete_rem

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()


        val_pid_index = [len(x) for x in self.pid_index.values()]
        min_v = min(val_pid_index)
        max_v = max(val_pid_index)
        hist_pid_index = [val_pid_index.count(x) for x in range(min_v, max_v+1)]
        num_print = 5
        for i, x in enumerate(range(min_v, min_v+min(len(hist_pid_index), num_print))):
            print('dataset histogram [bin:{}, cnt:{}]'.format(x, hist_pid_index[i]))
        print('...')
        print('dataset histogram [bin:{}, cnt:{}]'.format(max_v, val_pid_index.count(max_v)))

        val_pid_index_upper = []
        for x in val_pid_index:
            v_remain = x % self.num_instances
            if v_remain == 0:
                val_pid_index_upper.append(x)
            else:
                if self.delete_rem:
                    if x < self.num_instances:
                        val_pid_index_upper.append(x - v_remain + self.num_instances)
                    else:
                        val_pid_index_upper.append(x - v_remain)
                else:
                    val_pid_index_upper.append(x - v_remain + self.num_instances)

        total_images = sum(val_pid_index_upper)
        total_images = total_images - (total_images % self.batch_size) - self.batch_size # approax
        self.total_images = total_images



    def _get_epoch_indices(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.pid_index[pid]) # whole index for each ID
            if self.delete_rem:
                if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            else:
                if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                elif (len(idxs) % self.num_instances) != 0:
                    idxs.extend(np.random.choice(idxs, size=self.num_instances - len(idxs) % self.num_instances, replace=False))

            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(int(idx))
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        # batch_idxs_dict: dictionary, len(batch_idxs_dict) is len(pidx), each pidx, num_instance x k samples
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0: avai_pids.remove(pid)

        return final_idxs

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)
        # return iter(self._get_epoch_indices())

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            indices = self._get_epoch_indices()
            yield from indices

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        logger = logging.getLogger('reid.train')
        logger.info("start batch dividing.")
        t0 = time.time()
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        logger.info('batch divide time: {:.2f}s'.format(time.time()-t0))
        return iter(final_idxs)

    def __len__(self):
        return self.length
    

class DomainIdentitySampler(Sampler):
    """
    all ids are from one domain in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, num_pids):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_pids = num_pids

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        logger = logging.getLogger('reid.train')
        logger.info("All ids from the same domain in a batch. Start batch dividing.")
        t0 = time.time()
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        num_dom = len(self.num_pids)
        start, end = [0], [self.num_pids[0]]
        for i in range(num_dom-1):
            start.append(start[-1] + self.num_pids[i])
            end.append(start[-1] + self.num_pids[i+1])
        pids_domain_wise = [
            avai_pids[start[i]:end[i]]\
            for i in range(len(self.num_pids))
        ]
        remain_pids = []
        
        while len(pids_domain_wise) > 0:
            pids = random.choice(pids_domain_wise)
            ind = pids_domain_wise.index(pids)
            if len(pids) < self.num_pids_per_batch:
                remain_pids.extend(pids)
                pids_domain_wise.remove(pids)
                if len(remain_pids)>=self.num_pids_per_batch:
                    selected_pids = random.sample(remain_pids, self.num_pids_per_batch)
                else:
                    continue
            else:
                selected_pids = random.sample(pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    if pid in remain_pids:
                        remain_pids.remove(pid)
                        continue
                    pids_domain_wise[ind].remove(pid)
                    if len(pids_domain_wise[ind]) == 0:
                        pids_domain_wise.remove(pids)

        logger.info('batch divide time: {:.2f}s'.format(time.time()-t0))
        return iter(final_idxs)
    
    def __len__(self):
        return self.length


class HardNegetiveSampler(DomainIdentitySampler, GraphSampler):
    def __init__(self, cfg, centers, train_set, batch_size, num_pids, model, transform):
        self.cfg = cfg
        self.centers = centers
        self.num_classes = len(centers)

        GraphSampler.__init__(self, train_set.img_items, model, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_WORKERS, transform)

        self.data_source = train_set.img_items
        self.pid_dict = train_set.pid_dict
        self.batch_size = batch_size
        self.num_instances = cfg.DATALOADER.NUM_INSTANCE
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_pids = num_pids

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.epoch = 0
        self.save_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)

    def __iter__(self):
        self.epoch = self.epoch + 1
        # if self.epoch <= 10:
        #     return DomainIdentitySampler.__iter__(self)
        logger = logging.getLogger('reid.train')
        logger.info("start batch dividing.")
        logger.info("Hard Sampling based on centers.")
        t0 = time.time()

        
        #### use model to calc dist
        sam_index = []
        for pid in self.pids:
            # random select one image for each id
            index = np.random.choice(self.index_dic[pid], size=1)[0]
            sam_index.append(index)
        dataset = [self.data_source[i] for i in sam_index]
        dist_mat = GraphSampler.calc_distance(self, dataset)
        N = dist_mat.shape[0]
        mask = torch.eye(N,N, device=dist_mat.device) * 1e15
        dist_mat = dist_mat + mask
        #### use model to calc dist

        # #### use class centers to calc dist
        # centers = self.centers.detach()
        # dist_mat = euclidean_dist(centers, centers)
        # N = dist_mat.shape[0]
        # mask = torch.eye(N,N, device=dist_mat.device) * 1e15
        # dist_mat = dist_mat + mask
        # #### use class centers to calc dist

        num_k = self.batch_size // self.num_instances - 1
        _, topk_index = torch.topk(dist_mat.cuda(), num_k, largest=False)
        topk_index = topk_index.cpu().numpy()

        # ######## save results
        # if self.save_path is not None:  
        #     save_path = os.path.join(self.save_path, 'gs_results')
        #     filename = os.path.join(save_path, 'shs%d.json' % self.epoch)
        #     import json
        #     d = self.pid_dict
        #     ivd = {v: k for k, v in d.items()}
        #     save_dict = {ivd[i]: list(ivd[topk_index[i, j]] for j in range(num_k)) for i in range(self.num_classes)}
        #     # save_dict = {'topk':topk_index.tolist()}
        #     save_json = json.dumps(save_dict)
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     f = open(filename, 'w')
        #     f.write(save_json)
        #     f.close()
        #     print("sample results saved!!!")

        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            anchor_pid = random.choice(avai_pids)
            ind = self.pids.index(anchor_pid)
            selected_pids = list(topk_index[ind])
            selected_pids = [self.pids[i] for i in selected_pids]
            selected_pids.append(anchor_pid)
            remove = 0
            avai_pids_rest = copy.deepcopy(avai_pids)
            selected_pids_cp = copy.deepcopy(selected_pids)
            for p in selected_pids_cp:
                if p not in avai_pids:
                    selected_pids.remove(p)
                    remove += 1
                else:
                    avai_pids_rest.remove(p)
            add_pids = random.sample(avai_pids_rest, remove)
            del(avai_pids_rest)
            del(selected_pids_cp)
            selected_pids.extend(add_pids)
            # assert len(selected_pids) == self.num_pids_per_batch
            # selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        logger.info('batch divide time: {:.2f}s'.format(time.time()-t0))
        return iter(final_idxs)

    def __len__(self):
        return self.length


class DomainSuffleSampler(Sampler):

    def __init__(self, data_source: str, batch_size: int, num_instances: int, delete_rem: bool, seed: Optional[int] = None, cfg = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.delete_rem = delete_rem

        self.index_pid = defaultdict(list)
        self.pid_domain = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):

            domainid = info[3]['domains']
            if cfg.DATALOADER.CAMERA_TO_DOMAIN:
                pid = info[1] + str(domainid)
            else:
                pid = info[1]
            self.index_pid[index] = pid
            # self.pid_domain[pid].append(domainid)
            self.pid_domain[pid] = domainid
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.domains = list(self.pid_domain.values())

        self.num_identities = len(self.pids)
        self.num_domains = len(set(self.domains))

        self.batch_size //= self.num_domains
        self.num_pids_per_batch //= self.num_domains

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()


        val_pid_index = [len(x) for x in self.pid_index.values()]
        min_v = min(val_pid_index)
        max_v = max(val_pid_index)
        hist_pid_index = [val_pid_index.count(x) for x in range(min_v, max_v+1)]
        num_print = 5
        for i, x in enumerate(range(min_v, min_v+min(len(hist_pid_index), num_print))):
            print('dataset histogram [bin:{}, cnt:{}]'.format(x, hist_pid_index[i]))
        print('...')
        print('dataset histogram [bin:{}, cnt:{}]'.format(max_v, val_pid_index.count(max_v)))

        val_pid_index_upper = []
        for x in val_pid_index:
            v_remain = x % self.num_instances
            if v_remain == 0:
                val_pid_index_upper.append(x)
            else:
                if self.delete_rem:
                    if x < self.num_instances:
                        val_pid_index_upper.append(x - v_remain + self.num_instances)
                    else:
                        val_pid_index_upper.append(x - v_remain)
                else:
                    val_pid_index_upper.append(x - v_remain + self.num_instances)

        cnt_domains = [0 for x in range(self.num_domains)]
        for val, index in zip(val_pid_index_upper, self.domains):
            cnt_domains[index] += val
        self.max_cnt_domains = max(cnt_domains)
        self.total_images = self.num_domains * (self.max_cnt_domains - (self.max_cnt_domains % self.batch_size) - self.batch_size)



    def _get_epoch_indices(self):


        def _get_batch_idxs(pids, pid_index, num_instances, delete_rem):
            batch_idxs_dict = defaultdict(list)
            for pid in pids:
                idxs = copy.deepcopy(pid_index[pid])
                if delete_rem:
                    if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                else:
                    if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                    elif (len(idxs) % self.num_instances) != 0:
                        idxs.extend(np.random.choice(idxs, size=self.num_instances - len(idxs) % self.num_instances, replace=False))

                np.random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(int(idx))
                    if len(batch_idxs) == num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []
            return batch_idxs_dict

        batch_idxs_dict = _get_batch_idxs(self.pids, self.pid_index, self.num_instances, self.delete_rem)

        # batch_idxs_dict: dictionary, len(batch_idxs_dict) is len(pidx), each pidx, num_instance x k samples
        avai_pids = copy.deepcopy(self.pids)

        local_avai_pids = \
            [[pids for pids, idx in zip(avai_pids, self.domains) if idx == i]
             for i in list(set(self.domains))]
        local_avai_pids_save = copy.deepcopy(local_avai_pids)


        revive_idx = [False for i in range(self.num_domains)]
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch and not all(revive_idx):
            for i in range(self.num_domains):
                selected_pids = np.random.choice(local_avai_pids[i], self.num_pids_per_batch, replace=False)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)
                        local_avai_pids[i].remove(pid)
            for i in range(self.num_domains):
                if len(local_avai_pids[i]) < self.num_pids_per_batch:
                    print('{} is recovered'.format(i))
                    batch_idxs_dict_new = _get_batch_idxs(self.pids, self.pid_index, self.num_instances, self.delete_rem)

                    revive_idx[i] = True
                    cnt = 0
                    for pid, val in batch_idxs_dict_new.items():
                        if self.domains[cnt] == i:
                            batch_idxs_dict[pid] = copy.deepcopy(batch_idxs_dict_new[pid])
                        cnt += 1
                    local_avai_pids[i] = copy.deepcopy(local_avai_pids_save[i])
                    avai_pids.extend(local_avai_pids_save[i])
                    avai_pids = list(set(avai_pids))
        return final_idxs

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            indices = self._get_epoch_indices()
            yield from indices
