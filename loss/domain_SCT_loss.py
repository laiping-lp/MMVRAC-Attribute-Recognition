'''
copy from MetaBIN
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .triplet_loss import euclidean_dist, normalize, cosine_dist, cosine_sim

def domain_SCT_loss(embedding, domain_labels, norm_feat=False, type='cos_sim'):

    # eps=1e-05
    if norm_feat: embedding = normalize(embedding, axis=-1)
    unique_label = torch.unique(domain_labels)
    embedding_all = list()
    for i, x in enumerate(unique_label):
        embedding_all.append(embedding[x == domain_labels])
    num_domain = len(embedding_all)
    loss_all = []
    for i in range(num_domain):
        feat = embedding_all[i]
        center_feat = torch.mean(feat, 0)
        if type == 'euc':
            loss = torch.mean(euclidean_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cos':
            loss = torch.mean(cosine_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cos_sim':
            loss = torch.mean(cosine_sim(center_feat.view(1, -1), feat))
            loss_all.append(loss)

    loss_all = torch.mean(torch.stack(loss_all))

    return loss_all

def domain_shuffle_loss(dist_mat, labels, domains, scale=1.):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    #### same image | diff domain -> positive
    #### same domain & diff label -> negetive
    is_pos = domains.expand(N, N).ne(domains.expand(N, N).t())
    is_neg = domains.expand(N, N).eq(domains.expand(N, N).t())
    is_same_ids = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_same_image = torch.eye(N).bool().to(is_pos.device)
    is_pos = is_pos + is_same_image
    is_neg = is_neg * ~is_same_ids

    # is_pos = is_pos & ~same_ids
    # is_neg = is_neg & ~same_ids

    # dist_ap, relative_p_inds = torch.max(
    #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # dist_an, relative_n_inds = torch.min(
    #     dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # dist_ap = dist_ap.squeeze(1)
    # dist_an = dist_an.squeeze(1)

    dist_mat1, dist_mat2 = dist_mat.clone(), dist_mat.clone()
    dist_mat1[is_neg] = 0.
    dist_mat2[is_pos] = 1e12
    # dist_mat2[same_ids] = 1e12
    dist_ap,relative_p_inds = dist_mat1.max(1)
    dist_an, relative_n_inds = dist_mat2.min(1)

    y = dist_an.new().resize_as_(dist_an).fill_(1)
    loss = nn.SoftMarginLoss()((dist_an - dist_ap) / scale, y)

    return loss