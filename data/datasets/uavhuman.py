# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com

@author: Li Tianjiao
@contact: tianjiao_li@mymail.sutd.edu.sg
"""

import glob
import re
import json

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
        
        # # """Comment for Competition Splits
        self.query = query
        self.gallery = gallery
        # """
        backpack_attempt = False
        hat_attempt = False
        UCC_first_attempt = False
        UCC_second_attepmt = False
        LCC_attepmt = False

        if backpack_attempt:
            # backpack first_attempt
            file_path = "##{file_find_in_google_drive_link}##/ALL_best_model/attr_all_result_json_file/backpack_all.json"
            with open(file_path,'r') as f:
                load_data = json.load(f)
            new_load_data = []
            query_dir_for_test = []
            gallery_dir_for_test = []
            for data in load_data:
                if data['pre_label'] == 4:
                    query_dir_for_test.append(data)
                elif data['pre_label'] in [0,1,2,3]:
                    gallery_dir_for_test.append(data)
                else:
                    new_load_data.append(data)       
            query = self._process_dir_json(query_dir_for_test, is_train=False)
            gallery = self._process_dir_json(gallery_dir_for_test, is_train=False)
        
        if hat_attempt:
            # hat first attempt
            file_path = "##{file_find_in_google_drive_link}##/ALL_best_model/attr_all_result_json_file/hat_all.json"
            with open(file_path,'r') as f:
                load_data = json.load(f)
            new_load_data = []
            query_dir_for_test = []
            gallery_dir_for_test = []
            for data in load_data:
                if data['pre_label'] == 3:
                    query_dir_for_test.append(data)
                elif data['pre_label'] in [0,1,2,4]:
                    gallery_dir_for_test.append(data)
                else:
                    new_load_data.append(data)
            query = self._process_dir_json(query_dir_for_test, is_train=False)
            gallery = self._process_dir_json(gallery_dir_for_test, is_train=False)
        
        if UCC_first_attempt:
            # UCC first attempt
            file_path = "##{file_find_in_google_drive_link}##/ALL_best_model/attr_all_result_json_file/upper_color_all.json"
            with open(file_path,'r') as f:
                load_data = json.load(f)
            new_load_data = []
            query_dir_for_test = []
            gallery_dir_for_test = []
            for data in load_data:
                if data['pre_label'] == 1:
                    query_dir_for_test.append(data)
                elif data['pre_label'] in [5,2,10]:
                    gallery_dir_for_test.append(data)
                else:
                    new_load_data.append(data)
            train = self._process_dir_json(new_load_data,is_train=True)        
            query = self._process_dir_json(query_dir_for_test, is_train=False)
            gallery = self._process_dir_json(gallery_dir_for_test, is_train=False)
        
        if UCC_second_attepmt:
            # second attempt
            file_path = "##{file_find_in_google_drive_link}##/ALL_best_model/attr_with_reid/UCC_new1.json"
            with open(file_path,'r') as f:
                load_data = json.load(f)
            new_load_data = []
            query_dir_for_test = []
            gallery_dir_for_test = []
            for data in load_data:
                if data['pre_label'] == 1:
                    query_dir_for_test.append(data)
                elif data['pre_label'] in [7,10]:
                    gallery_dir_for_test.append(data)
                else:
                    new_load_data.append(data)
            train = self._process_dir_json(new_load_data,is_train=True)        
            query = self._process_dir_json(query_dir_for_test, is_train=False)
            gallery = self._process_dir_json(gallery_dir_for_test, is_train=False)

        if LCC_attepmt:
            # LCC first_attempt
            file_path = "##{file_find_in_google_drive_link}##/ALL_best_model/attr_all_result_json_file/lower_color_all.json"
            with open(file_path,'r') as f:
                load_data = json.load(f)
            new_load_data = []
            query_dir_for_test = []
            gallery_dir_for_test = []
            for data in load_data:
                if data['pre_label'] == 9:
                    query_dir_for_test.append(data)
                elif data['pre_label'] in [2,7]:
                    gallery_dir_for_test.append(data)
                else:
                    new_load_data.append(data)
            train = self._process_dir_json(new_load_data,is_train=True)        
            query = self._process_dir_json(query_dir_for_test, is_train=False)
            gallery = self._process_dir_json(gallery_dir_for_test, is_train=False)
             

        self.train = train
        self.query = query
        self.gallery = gallery
        # import ipdb;ipdb.set_trace()

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        
        # """Comment for Competition Splits
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        # """
        # super(UAVHuman, self).__init__(train, query, gallery,q_pre_label,q_real_label,g_pre_label,g_real_label, **kwargs)
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

    def _process_dir_json(self,json_list,is_train=True):
        img_paths = []
        pre_label = []
        real_label = []
        # pid and camid patterns
        pattern_pid = re.compile(r'P([-\d]+)S([-\d]+)')
        pattern_camid = re.compile(r'A([-\d]+)R([-\d])_([-\d]+)_([-\d]+)')
        distractor_pid = 50000
        for data in json_list:
            img_paths.append(data['file_path'])
            pre_label.append(data['pre_label'])
            real_label.append(data['real_label'])
        pid_container = set()
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            if fname.startswith('D'):
                # continue
                pid = int(distractor_pid)
            else:
                pid_part1, pid_part2 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2)
            
            if pid == -1: continue  # junk images are just ignored
            if pid == 3109 or pid == 8405: 
                import ipdb; ipdb.set_trace()
                continue

            pid_container.add(pid)
        dataset = []
        for (img_path,pre,real) in zip(img_paths,pre_label,real_label):
            fname = osp.split(img_path)[-1]
            if fname.startswith('D'):
                # continue
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
            else:
                pid = self.dataset_name + "_" + str(pid)
            attributes = {
                "pre_label": pre,
                "real_label": real,
            }
            # attributes = None
            dataset.append((img_path, pid, camid, attributes))
        return dataset
            
            

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
                # continue
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
        UCC_number_list = [0] * 12
        LCC_number_list = [0] * 12

        
        set_UCC_number = False
        set_LCC_number = False
        set_LCC_number_1 = False      
        # set_UCC_number = True
        # set_LCC_number = True
        # set_LCC_number_1 = True
        dataset = []
        for img_path in img_paths:
            fname = osp.split(img_path)[-1]
            if fname.startswith('D'):
                # continue
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
                gender = int(pattern_gender.search(fname).groups()[0][0]) - 1 # 0: n/a; 1: male; 2: female
                
                backpack = int(pattern_backpack.search(fname).groups()[0][0])
                if backpack == 5:
                    backpack = 0 # 0: n/a; 1: red; 2: black; 3: green; 4: yellow; 5: n/a
                hat = int(pattern_hat.search(fname).groups()[0][0]) # 0: n/a; 1: red; 2: black; 3: yellow; 4: white; 5: n/a
                if hat == 5:
                    hat = 0
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
                # control LCC number for LCC
                if set_LCC_number_1:
                    if(lower_color == 6): # grey 100_1
                        if(LCC_number_list[lower_color] > 100):
                            continue
                        else:
                            LCC_number_list[lower_color] += 1
                    else:
                        LCC_number_list[lower_color] += 1
                # control UCC number for UCC and UCS and LCS
                if set_UCC_number:
                    if(upper_color == 7): # white 1000_1
                        if(UCC_number_list[upper_color] > 1000):
                            continue
                        else:
                            UCC_number_list[upper_color] += 1
                    elif upper_color == 3: # blue 100_1
                        if(UCC_number_list[upper_color] > 100):
                            continue
                        else:
                            UCC_number_list[upper_color] += 1
                    elif upper_color == 6: # grey * 2_1 *3_5
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        dataset.append((img_path, pid, camid, attributes))
                    elif upper_color == 1: # red * 4_1 * 8_3 *10_4 *8_5
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                    elif upper_color == 2: # black 2000_1  1000_3 1000_5
                        if(UCC_number_list[upper_color] > 2000):
                            continue
                        else:
                            UCC_number_list[upper_color] += 1
                    elif upper_color == 4: # green 100_2
                        if(UCC_number_list[upper_color] > 100):
                            continue
                        else:
                            UCC_number_list[upper_color] += 1
                    elif upper_color == 11: # pink * 5_2
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                    else:
                        UCC_number_list[upper_color] += 1
                # control LCC number for hat
                if set_LCC_number: 
                    if(lower_color == 6): # grey 100_1
                        if(LCC_number_list[lower_color] > 100):
                            continue
                        else:
                            LCC_number_list[lower_color] += 1
                    elif(lower_color == 2): # black 5000
                        if(LCC_number_list[lower_color] > 5000):
                            continue
                        else:
                            LCC_number_list[lower_color] += 1
                    elif(lower_color == 9): # dark brown * 3
                        LCC_number_list[lower_color] += 1
                        LCC_number_list[lower_color] += 1
                        LCC_number_list[lower_color] += 1
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                    elif(lower_color == 3): # blue * 2_5
                        LCC_number_list[lower_color] += 1
                        LCC_number_list[lower_color] += 1
                        dataset.append((img_path, pid, camid, attributes))
                    else:
                        LCC_number_list[lower_color] += 1

            else:
                pid = self.dataset_name + "_" + str(pid)

                # attributes infos
                gender = int(pattern_gender.search(fname).groups()[0][0]) - 1 # 0: n/a; 1: male; 2: female
                backpack = int(pattern_backpack.search(fname).groups()[0][0])
                if backpack == 5:
                    backpack = 0 # 0: n/a; 1: red; 2: black; 3: green; 4: yellow; 5: n/a
                hat = int(pattern_hat.search(fname).groups()[0][0]) # 0: n/a; 1: red; 2: black; 3: yellow; 4: white; 5: n/a
                if hat == 5:
                    hat = 0
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
            # if relabel: pid = pid2label[pid] # relabel in common.py
            dataset.append((img_path, pid, camid, attributes))
        if is_train:
            print(f"UCC_number_list: {UCC_number_list}, sum: {sum(UCC_number_list)}")
            print(f"LCC_number_list: {LCC_number_list}, sum: {sum(LCC_number_list)}")
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