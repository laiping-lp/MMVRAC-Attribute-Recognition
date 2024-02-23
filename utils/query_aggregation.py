import numpy as np


def query_aggregate(distmat, q_pids):
    print('=> Enter query aggregation')
    uniq_ids = np.unique(q_pids)
    for pid in uniq_ids:
        indexs = np.argwhere(q_pids==pid).squeeze()
        avg_dist = np.mean(distmat[indexs], axis=0)
        distmat[indexs] = avg_dist

    return distmat