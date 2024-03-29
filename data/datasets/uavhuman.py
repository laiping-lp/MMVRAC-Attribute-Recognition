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
        # self.query = query
        # self.gallery = gallery
        # """
        # backpack first_attempt
        # file_path = "/data3/laiping/recurrence/ALL_best_model/attr_all_result_json_file/backpack.json"
        # with open(file_path,'r') as f:
        #     load_data = json.load(f)
        # new_load_data = []
        # query_dir_for_test = []
        # gallery_dir_for_test = []
        # for data in load_data:
        #     if data['pre_label'] == 4:
        #         query_dir_for_test.append(data)
        #     # elif data['pre_label'] in [0,3]:
        #     elif data['pre_label'] in [0,1,2,3]:
        #         gallery_dir_for_test.append(data)
        #     else:
        #         new_load_data.append(data)
        # second_attempt
        # file_path = "/data3/laiping/recurrence/ALL_best_model/attr_all_result_json_file/backpack_after_yellow.json"
        # with open(file_path,'r') as f:
        #     load_data = json.load(f)
        # new_load_data = []
        # query_dir_for_test = []
        # gallery_dir_for_test = []
        # for data in load_data:
        #     if data['pre_label'] == 3:
        #         query_dir_for_test.append(data)
        #     elif data['pre_label'] == 0:
        #         gallery_dir_for_test.append(data)
        #     else:
        #         new_load_data.append(data)
        
        # UCC first attempt
        # file_path = "/data3/laiping/recurrence/ALL_best_model/attr_all_result_json_file/upper_color.json"
        # with open(file_path,'r') as f:
        #     load_data = json.load(f)
        # new_load_data = []
        # query_dir_for_test = []
        # gallery_dir_for_test = []
        # for data in load_data:
        #     if data['pre_label'] == 1:
        #         query_dir_for_test.append(data)
        #     # elif data['pre_label'] in [0,3]:
        #     # elif data['pre_label'] in [0,2,3,4,5,6,7,8,9,10,11]:
        #     elif data['pre_label'] in [5,2,10]:
        #         gallery_dir_for_test.append(data)
        #     else:
        #         new_load_data.append(data)
        
        # second attempt
        file_path = "/data3/laiping/recurrence/ALL_best_model/attr_all_result_json_file/UCC_after_red.json"
        with open(file_path,'r') as f:
            load_data = json.load(f)
        new_load_data = []
        query_dir_for_test = []
        gallery_dir_for_test = []
        for data in load_data:
            if data['pre_label'] == 1:
                query_dir_for_test.append(data)
            # elif data['pre_label'] in [0,3]:
            # elif data['pre_label'] in [0,2,3,4,5,6,7,8,9,10,11]:
            elif data['pre_label'] in [7,10]:
                gallery_dir_for_test.append(data)
            else:
                new_load_data.append(data)

        # LCC first_attempt
        # file_path = "/data3/laiping/recurrence/ALL_best_model/attr_all_result_json_file/lower_color.json"
        # with open(file_path,'r') as f:
        #     load_data = json.load(f)
        # new_load_data = []
        # query_dir_for_test = []
        # gallery_dir_for_test = []
        # for data in load_data:
        #     if data['pre_label'] == 9:
        #         query_dir_for_test.append(data)
        #     # elif data['pre_label'] in [0,3]:
        #     elif data['pre_label'] in [2,7]:
        #         gallery_dir_for_test.append(data)
        #     else:
        #         new_load_data.append(data)

        # train = self._process_dir_json(new_load_data,is_train=True)        
        # query = self._process_dir_json(query_dir_for_test, is_train=False)
        # gallery = self._process_dir_json(gallery_dir_for_test, is_train=False)

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
        gender_number_list = [0] * 2
        backpack_number_list = [0] * 5
        hat_number_list = [0] * 5
        UCC_number_list = [0] * 12
        UCS_number_list = [0] * 4
        LCC_number_list = [0] * 12
        LCS_number_list = [0] * 4

        set_gender_number = False
        set_Backpack_number = False
        set_Hat_number = False
        set_UCC_number = False
        set_UCS_number = False
        set_LCC_number = False
        set_LCS_number = False

        # set_gender_number = True
        # set_Backpack_number = True
        # set_Hat_number = True
        set_UCC_number = True
        # set_UCS_number = True
        # set_LCC_number = True
        # set_LCS_number = True
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
                # control gender male and female rate 
                if set_gender_number:
                    if(gender == 0):
                        if(gender_number_list[gender] > 1288):
                            continue
                        else:
                            gender_number_list[gender] += 1
                    else:
                        gender_number_list[gender] += 1
                # control backpack number
                if set_Backpack_number:
                    if(backpack == 0): # n/a number 4000_1  3000_2  2000_3 1000_4
                        if(backpack_number_list[backpack] > 500):
                            continue
                        else:
                            backpack_number_list[backpack] += 1
                    elif(backpack == 4): # yellow number * 2_1 *3_2 *4_5 *5_6
                        backpack_number_list[backpack] += 1
                        backpack_number_list[backpack] += 1
                        backpack_number_list[backpack] += 1
                        backpack_number_list[backpack] += 1
                        backpack_number_list[backpack] += 1
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                    elif(backpack == 3): # green number * 2_3 *3_5
                        backpack_number_list[backpack] += 1
                        backpack_number_list[backpack] += 1
                        # backpack_number_list[backpack] += 1
                        # dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                    else:
                        backpack_number_list[backpack] += 1
                # control hat number
                if set_Hat_number:
                    if(hat == 0):
                        if(hat_number_list[hat] > 3000):
                            continue
                        else:
                            hat_number_list[hat] += 1
                    else:
                        hat_number_list[hat] += 1
                # control UCS number
                if set_UCS_number:
                    if(upper_style == 1): # long * 2
                        UCS_number_list[upper_style] += 1
                        UCS_number_list[upper_style] += 1
                        dataset.append((img_path, pid, camid, attributes))   
                    else:
                        UCS_number_list[upper_style] += 1
                # control LCS number
                if set_LCS_number:
                    if(lower_style == 1): # long 5000
                        if(LCS_number_list[lower_style] > 5000):
                            continue
                        else:
                            LCS_number_list[lower_style] += 1
                    else:
                        LCS_number_list[lower_style] += 1
                # control UCC number
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
                        UCC_number_list[upper_color] += 1
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                    elif upper_color == 1: # red * 4_1 * 8_3 *10_4 *8_5
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        # UCC_number_list[upper_color] += 1
                        # UCC_number_list[upper_color] += 1
                        # UCC_number_list[upper_color] += 1
                        # UCC_number_list[upper_color] += 1
                        # UCC_number_list[upper_color] += 1
                        # UCC_number_list[upper_color] += 1
                        # dataset.append((img_path, pid, camid, attributes))
                        # dataset.append((img_path, pid, camid, attributes))
                        # dataset.append((img_path, pid, camid, attributes))
                        # dataset.append((img_path, pid, camid, attributes))
                        # dataset.append((img_path, pid, camid, attributes))
                        # dataset.append((img_path, pid, camid, attributes))
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
                        UCC_number_list[upper_color] += 1
                        UCC_number_list[upper_color] += 1
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                    elif upper_color == 5: # multicolor 200_3
                        if(UCC_number_list[upper_color] > 200):
                            continue
                        else:
                            UCC_number_list[upper_color] += 1
                    else:
                        UCC_number_list[upper_color] += 1
                # control LCC number
                if set_LCC_number:
                    if(lower_color == 6): # grey 100_1
                        if(LCC_number_list[lower_color] > 100):
                            continue
                        else:
                            LCC_number_list[lower_color] += 1
                    # elif(lower_color == 2): # black 3000_2 5000_6
                    #     if(LCC_number_list[lower_color] > 5000):
                    #         continue
                    #     else:
                    #         LCC_number_list[lower_color] += 1
                    elif(lower_color == 9): # dark brown * 2_2 *3_3 *4_4 *3_5
                        LCC_number_list[lower_color] += 1
                        LCC_number_list[lower_color] += 1
                        LCC_number_list[lower_color] += 1
                        LCC_number_list[lower_color] += 1
                        LCC_number_list[lower_color] += 1
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                        dataset.append((img_path, pid, camid, attributes))
                    elif(lower_color == 3): # blue * 2_5
                        LCC_number_list[lower_color] += 1
                        LCC_number_list[lower_color] += 1
                        # LCC_number_list[lower_color] += 1
                        # LCC_number_list[lower_color] += 1
                        # dataset.append((img_path, pid, camid, attributes))
                        # dataset.append((img_path, pid, camid, attributes))
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
            print(f"Gender_number_list: {gender_number_list}, sum: {sum(gender_number_list)}")
            print(f"Backpack_number_list: {backpack_number_list}, sum: {sum(backpack_number_list)}")
            print(f"Hat_number_list: {hat_number_list}, sum: {sum(hat_number_list)}")
            print(f"UCC_number_list: {UCC_number_list}, sum: {sum(UCC_number_list)}")
            print(f"UCS_number_list: {UCS_number_list}, sum: {sum(UCS_number_list)}")
            print(f"LCC_number_list: {LCC_number_list}, sum: {sum(LCC_number_list)}")
            print(f"LCS_number_list: {LCS_number_list}, sum: {sum(LCS_number_list)}")
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