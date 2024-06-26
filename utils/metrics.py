from re import T
from time import time
import torch
import numpy as np
import os
from utils.reranking import re_ranking
import json
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from utils.faiss_rerank import compute_jaccard_distance
import re
import os

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()

def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1 - cosine

def cosine_sim(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def eval_func_make_new_json(distmat, max_rank=50, query=None, gallery=None, log_path=None,gen_result=False,other_data=None):
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    new_load_data = []
    for data in other_data:
        new_data = {
            "file_path":"",
            "pre_label":"",
            "real_label":"" 
        }
        new_data["file_path"] = data[0]
        new_data['pre_label'] = data[3]['pre_label']
        new_data['real_label'] = data[3]['real_label']
        new_load_data.append(new_data)
    for data in query:
        new_data = {
            "file_path":"",
            "pre_label":"",
            "real_label":"" 
        }
        new_data["file_path"] = data[0]
        new_data['pre_label'] = data[3]['pre_label']
        new_data['real_label'] = data[3]['real_label']
        new_load_data.append(new_data)
    index_flag = [1] * len(gallery)
    set_backpack_parameter = False
    set_hat_parameter = False
    set_UCC_parameter_0 = False
    set_UCC_parameter_1 = False
    set_LCC_parameter = False
    if set_backpack_parameter:
        index_range = 200
        threshold_value = 0.3
    if set_hat_parameter:
        index_range = 5
        threshold_value = 0.3
    if set_UCC_parameter_0:
        index_range = 1000
        threshold_value = 1.1
    if set_UCC_parameter_1:
        index_range = 5
        threshold_value = 0.4
    if set_LCC_parameter:
        index_range = 30
        threshold_value = 0.5
    for i,(query_,index) in enumerate(zip(query,indices)):
        new_data = {
            "file_path":"",
            "pre_label":"",
            "real_label":"" 
        }
        for index_ in index[:index_range]: 
            if(distmat[i][index_]) < threshold_value: 
    
                if index_flag[index_] ==  1:
                    new_data["file_path"] = gallery[index_][0]
                    new_data['pre_label'] = query_[3]['pre_label']
                    new_data['real_label'] = gallery[index_][3]['real_label']
                    new_load_data.append(new_data)
                    index_flag[index_] = 0
                elif index_flag[index_] == 0:
                    continue
    for i in range(len(gallery)):
        new_data = {
            "file_path":"",
            "pre_label":"",
            "real_label":"" 
        }
        if(index_flag[i] == 1):
            new_data["file_path"] = gallery[i][0]
            new_data['pre_label'] = gallery[i][3]['pre_label']
            new_data['real_label'] = gallery[i][3]['real_label']
            new_load_data.append(new_data) 
           
    print(len(new_load_data))
    acc_count = 0
    for data in new_load_data:
        if(data['pre_label'] == data['real_label']):
            acc_count += 1
            
    print("The new accuracy: ",acc_count * 100 / len(new_load_data))
    # generate attr_reid json file
    if set_backpack_parameter:
        with open("##{your_folder}##/ALL_best_model/attr_with_reid/Backpack_new.json",'w') as f_1:
            json.dump(new_load_data,f_1)
    if set_hat_parameter:
        with open("##{your_folder}##/ALL_best_model/attr_with_reid/Hat_new.json",'w') as f_1:
            json.dump(new_load_data,f_1)
    if set_LCC_parameter:
        with open("##{your_folder}##/ALL_best_model/attr_with_reid/LCC_new.json",'w') as f_1:
            json.dump(new_load_data,f_1)
    if set_UCC_parameter_0:
        with open("##{your_folder}##/ALL_best_model/attr_with_reid/UCC_new1.json",'w') as f_1:
            json.dump(new_load_data,f_1)
    if set_UCC_parameter_1:
        with open("##{your_folder}##/ALL_best_model/attr_with_reid/UCC_new2.json",'w') as f_1:
            json.dump(new_load_data,f_1)
    

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, query=None, gallery=None, log_path=None,gen_result=False):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    gen_result = gen_result
    if gen_result:
        # save query result
        file_path =  make_case_json(g_pids, query, gallery, indices, matches, log_path)
        # sort AP in ascending order, get last 2% data to analyse
        # AP_indices = np.argsort(all_AP) 
        # root_dir = log_path
        # gen_bad_case_img(root_dir, file_path, AP_indices, all_AP)
        # # random select case to analyse
        # np.random.seed(6)
        # random_AP_indices = np.random.permutation(AP_indices)
        # gen_random_case_img(root_dir,file_path,random_AP_indices,all_AP)
        print("done!")

    return all_cmc, mAP

def make_case_json(g_pids, query, gallery,indices,matches, log_path):
    cases_data = []
    for (indexs,match,query) in tqdm(zip(indices,matches,query)):
        case = {
            "query_path" : "",
            "query_id": "", 
            "top_50_results_ids" : "",
            "top_50_results_matches": "" ,
            "top_50_results_path": ""            
            }
        case["query_path"] = query[0]
        case["query_id"] = query[1]
        gid = []
        gallery_path = []
        top_50_results_matches = []
        for i in indexs:
            gid.append(int(g_pids[i]))
            gallery_path.append(gallery[i])
        top_50_results_ids = []
        top_50_results_ids = gid[:50]
        top_50_results_path = []
        for path,match_ in zip(gallery_path,match[:50]):
            top_50_results_path.append(path[0])
            top_50_results_matches.append(int(match_))
        # top_50_results_matches = match[:50]
        case["top_50_results_ids"] = top_50_results_ids
        case["top_50_results_matches"] = top_50_results_matches
        case["top_50_results_path"] = top_50_results_path
        # print(case)
        cases_data.append(case)
        # break
    file_path = os.path.join(log_path, "cases.json")
    with open(file_path,"w") as f:
        json.dump(cases_data,f)
    
    return file_path
    
def gen_bad_case_img(root_dir, file_path, AP_indices, all_AP):
    num_to_extract = int(len(AP_indices)* 0.05)
    with open(file_path,'r') as f:
        load_data = json.load(f)
    save_path_folder = root_dir + "/bad_cases" 
    # folder exists delete then bulid
    if os.path.exists(save_path_folder):
        shutil.rmtree(save_path_folder)
        os.makedirs(save_path_folder)
    # bulid
    else:
        os.makedirs(save_path_folder)
    for idx in range(num_to_extract):
        # print(idx)
        query_idx = AP_indices[idx]
        # print("AP",all_AP[query_idx])
        query_path = load_data[query_idx]["query_path"]
        # print("query_path:",query_path)
        matchs = []
        matchs = load_data[query_idx]["top_50_results_matches"]
        result_path = []
        result_path.append(query_path)
        for i,path in enumerate(load_data[query_idx]["top_50_results_path"]):
            if(i == 10):
                break
            result_path.append(path)
        # print(result_path)
        # print(matchs[:10])
        fig, axs = plt.subplots(1,11, figsize=(12,3))
        for j,image_path in enumerate(result_path):
            image = Image.open(image_path)
            # show image in subplots
            axs[j].imshow(image)
            axs[j].axis('off')
            if(j==0):
                axs[j].set_title("query",color = 'green')
            elif matchs[j-1]:
                axs[j].set_title("True",color = 'green')
            else:
                axs[j].set_title("False",color = 'red')
        plt.tight_layout()
        # plt.show()
        save_path = os.path.join(save_path_folder,os.path.basename(query_path))
        plt.savefig(save_path)
        plt.close()
        # break
        
def gen_random_case_img(root_dir,file_path,random_AP_indices, all_AP):
    num_to_extract = int(len(random_AP_indices)* 0.10)
    with open(file_path,'r') as f:
        load_data = json.load(f)
    save_path_folder = root_dir + "/random_cases" 
    # folder exists delete then bulid
    if os.path.exists(save_path_folder):
        shutil.rmtree(save_path_folder)
        os.makedirs(save_path_folder)
    # bulid
    else:
        os.makedirs(save_path_folder)
    for idx in range(num_to_extract):
        # print(idx)
        query_idx = random_AP_indices[idx]
        # print("AP",all_AP[query_idx])
        query_path = load_data[query_idx]["query_path"]
        # print("query_path:",query_path)
        matchs = []
        matchs = load_data[query_idx]["top_50_results_matches"]
        result_path = []
        result_path.append(query_path)
        for i,path in enumerate(load_data[query_idx]["top_50_results_path"]):
            if(i == 10):
                break
            result_path.append(path)
        # print(result_path)
        # print(matchs[:10])
        fig, axs = plt.subplots(1,11, figsize=(12,3))
        for j,image_path in enumerate(result_path):
            image = Image.open(image_path)
            # show image in subplots
            axs[j].imshow(image)
            axs[j].axis('off')
            if(j==0):
                axs[j].set_title("query",color = 'green')
            elif matchs[j-1]:
                axs[j].set_title("True",color = 'green')
            else:
                axs[j].set_title("False",color = 'red')
        plt.tight_layout()
        # plt.show()
        save_path = os.path.join(save_path_folder,os.path.basename(query_path))
        plt.savefig(save_path)
        plt.close()
        # break
    


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50,  feat_norm=True, reranking=False, query_aggregate=False, feature_aggregate=False, query=None, gallery=None, log_path=None,gen_result=False,other_data=None):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.query = query
        self.gallery = gallery
        self.other_data = other_data
        self.log_path = log_path
        if feat_norm:
            print("The test feature is normalized")
        self.reranking = reranking
        self.query_aggregate = query_aggregate
        self.feature_aggregate = feature_aggregate
        self.gen_result = gen_result

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        if self.feature_aggregate:
            qf = feat_aggregate(qf, q_pids)
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        # import ipdb;ipdb.set_trace()

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            distmat = re_ranking(qf, gf, k1=4, k2=4, lambda_value=0.45)
            # distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_dist(qf, gf)
        if self.query_aggregate:
            distmat = query_aggregate(distmat, q_pids)
        # features, _ = extract_features(model, cluster_loader, print_freq=50)
        # features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        # rerank_dist = compute_jaccard_distance(feats, k1=4, k2=4)
        # eval_func_make_new_json_1(rerank_dist,num_query = self.num_query , query=self.query, gallery=self.gallery, log_path=self.log_path,gen_result=self.gen_result,other_data= self.other_data)
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, query=self.query, gallery=self.gallery, log_path=self.log_path,gen_result=self.gen_result)
        # new method for attribute recognition
        eval_func_make_new_json(distmat,query=self.query, gallery=self.gallery, log_path=self.log_path,gen_result=self.gen_result,other_data= self.other_data)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
    
    

class R1_mAP_eval_ensemble():
    def __init__(self, num_query, max_rank=50,  feat_norm=True, reranking=False, query_aggregate=False, feature_aggregate=False, query=None, gallery=None, log_path=None,gen_result=False, num_models=1):
        super(R1_mAP_eval_ensemble, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.query = query
        self.gallery = gallery
        self.log_path = log_path
        if feat_norm:
            print("The test feature is normalized")
        self.reranking = reranking
        self.query_aggregate = query_aggregate
        self.feature_aggregate = feature_aggregate
        self.gen_result = gen_result
        self.num_models = num_models

    def reset(self):
        self.feats = [[] for _ in range(self.num_models)]
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feats, pid, camid = output
        for i, feat in enumerate(feats):
            self.feats[i].append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        # import ipdb
        # ipdb.set_trace()
        feats = [torch.cat(feat, dim=0) for feat in self.feats]
        distmats = []
        for feat in feats:
            if self.feat_norm:
                feat = torch.nn.functional.normalize(feat, dim=1, p=2)  # along channel
            # query
            qf = feat[:self.num_query]
            if self.feature_aggregate:
                qf = feat_aggregate(qf, q_pids)
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            # gallery
            gf = feat[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])

            g_camids = np.asarray(self.camids[self.num_query:])
            
            if self.reranking:
                distmat = re_ranking(qf, gf, k1=4, k2=4, lambda_value=0.45)
                # distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
            else:
                # print('=> Computing DistMat with euclidean_distance')
                distmat = euclidean_dist(qf, gf)
                
            ##### key operation of ensemble
            distmats.append(distmat)
        ##### key operation of ensemble 
        distmat = np.mean(distmats, axis=0)
        # distmat = 0.9*distmats[0] + 0.1*distmats[1]
        
        
        if self.query_aggregate:
            distmat = query_aggregate(distmat, q_pids)
            
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, query=self.query, gallery=self.gallery, log_path=self.log_path,gen_result=self.gen_result)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



def query_aggregate(distmat, q_pids):
    print('=> Enter query aggregation')
    uniq_ids = np.unique(q_pids)
    for pid in uniq_ids:
        indexs = np.argwhere(q_pids==pid).squeeze()
        avg_dist = np.mean(distmat[indexs], axis=0)
        distmat[indexs] = avg_dist

    return distmat

def feat_aggregate(qf, q_pids):
    print('=> feature aggregation')
    uniq_ids = np.unique(q_pids)
    for pid in uniq_ids:
        indexs = np.argwhere(q_pids==pid).squeeze()
        avg_feat = np.mean(qf[indexs], axis=0)
        qf[indexs] = avg_feat

    return qf