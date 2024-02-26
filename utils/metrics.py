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


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, query=None, gallery=None, log_path=None):
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
    
    # bad_case analyse
    # file_path =  make_case_json(g_pids, query, gallery, indices, matches, log_path)
    # AP_indices = np.argsort(all_AP) # sort in ascending order
    # root_dir = log_path
    # gen_bad_case_img(root_dir, file_path, AP_indices, all_AP)


    return all_cmc, mAP

def make_case_json(g_pids, query, gallery,indices,matches, log_path):
    cases_data = []
    for indexs,match,query in tqdm(zip(indices,matches,query)):
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
    num_to_extract = int(len(AP_indices)* 0.02)
    with open(file_path,'r') as f:
        load_data = json.load(f)
    
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
        save_path_folder = root_dir + "/bad_cases" 
        if not os.path.exists(save_path_folder):
            os.makedirs(save_path_folder)
        save_path = os.path.join(save_path_folder,os.path.basename(query_path))
        plt.savefig(save_path)
        # break
        



class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50,  feat_norm=True, reranking=False, query_aggregate=False, feature_aggregate=False, query=None, gallery=None, log_path=None):
        super(R1_mAP_eval, self).__init__()
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

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            distmat = re_ranking(qf, gf, k1=4, k2=4, lambda_value=0.45)
            # distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
            if self.query_aggregate:
                distmat = query_aggregate(distmat, q_pids)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_dist(qf, gf)
            if self.query_aggregate:
                distmat = query_aggregate(distmat, q_pids)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, query=self.query, gallery=self.gallery, log_path=self.log_path)

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