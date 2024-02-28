# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com

@author: Li Tianjiao
@contact: tianjiao_li@mymail.sutd.edu.sg
"""

import glob
import re

import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class UAVHuman(ImageDataset):
    dataset_dir = "uavhuman-reid"
    dataset_name = "uavhuman"
    
    def __init__(self, root='./data', verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        
        # """Comment for Competition Splits
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        # """

        self._check_before_run()

        train = self._process_dir(self.train_dir, is_train=True)
        
        # """Comment for Competition Splits
        query = self._process_dir(self.query_dir, is_train=False)
        gallery = self._process_dir(self.gallery_dir, is_train=False)
        # """

        # if verbose:
        #     print("=> UAVHuman loaded")
            
        #     """Comment for Competition Split
        #     self.print_dataset_statistics(train, query, gallery)
        #     """
            
        #     self.print_dataset_statistics_for_train_only(train)

        self.train = train
        
        # """Comment for Competition Splits
        self.query = query
        self.gallery = gallery
        # """

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        
        # """Comment for Competition Splits
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        # """
        super(UAVHuman, self).__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

        """Comment for Competition Splits
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        """

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        # pid and camid patterns
        pattern_pid = re.compile(r'P([-\d]+)S([-\d]+)')
        pattern_camid = re.compile(r'A([-\d]+)R([-\d])_([-\d]+)_([-\d]+)')

        # attributes patterns
        pattern_gender = re.compile(r'G([-\d]+)')
        pattern_backpack = re.compile(r'B([-\d]+)')
        pattern_hat = re.compile(r'H([-\d]+)')
        pattern_upper = re.compile(r'UC([-\d]+)')
        pattern_lower = re.compile(r'LC([-\d]+)')
        pattern_action = re.compile(r'A([-\d]+)')

        distractor_pid = 50000

        pid_container = set()
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            if fname.startswith('D'):
                pid = int(distractor_pid)
            else:
                pid_part1, pid_part2 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2)
            
            if pid == -1: continue  # junk images are just ignored
            if pid == 3109 or pid == 8405: 
                import ipdb; ipdb.set_trace()
                continue

            pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            if fname.startswith('D'):
                pid = int(distractor_pid)
                camid = int(fname[-13:-8])
            else:
                pid_part1, pid_part2 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2)
                camid_part1, _, _, camid_part2 = pattern_camid.search(fname).groups()
                camid = int(camid_part1 + camid_part2)
            if pid == -1: continue  # junk images are just ignored
            if is_train:
                pid = self.dataset_name + "_" + str(pid)

                # attributes infos
                gender = int(pattern_gender.search(fname).groups()[0][0]) # 0: n/a; 1: male; 2: female
                backpack = int(pattern_backpack.search(fname).groups()[0][0]) # 0: n/a; 1: red; 2: black; 3: green; 4: yellow; 5: n/a
                hat = int(pattern_hat.search(fname).groups()[0][0]) # 0: n/a; 1: red; 2: black; 3: yellow; 4: white; 5: n/a
                upper_cloth = pattern_upper.search(fname).groups()[0]
                upper_color = int(upper_cloth[:2]) # 0: n/a; 1: red; 2: black; 3: blue; 4: green; 5: multicolor; 6: grey; 7: white; 8: yellow; 9: dark brown; 10: purple; 11: pink
                upper_style = int(upper_cloth[2]) # 0: n/a; 1: long; 2: short; 3: skirt
                lower_cloth = pattern_lower.search(fname).groups()[0]
                lower_color = int(lower_cloth[:2]) # 0: n/a; 1: red; 2: black; 3: blue; 4: green; 5: multicolor; 6: grey; 7: white; 8: yellow; 9: dark brown; 10: purple; 11: pink
                lower_style = int(lower_cloth[2]) # 0: n/a; 1: long; 2: short; 3: skirt
                action = int(pattern_action.search(fname).groups()[0])
                attributes = {
                    "gender": gender,
                    "backpack": backpack,
                    "hat": hat,
                    "upper_color": upper_color,
                    "upper_style": upper_style,
                    "lower_color": lower_color,
                    "lower_style": lower_style
                }
            else:
                attributes = None
            # if relabel: pid = pid2label[pid] # relabel in common.py
            dataset.append((img_path, pid, camid, attributes))

        return dataset

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams