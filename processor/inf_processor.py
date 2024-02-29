import logging
import time

import torch
import torch.nn as nn
from data.build_DG_dataloader import build_reid_test_loader

from utils.metrics import R1_mAP_eval
from utils.attribute_recognition import Attribute_Recognition

from prettytable import PrettyTable
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import os
import json

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 reranking=False,
                 iflog=True,
                 query=None,
                 gallery=None,
                 gen_result=False,
                 query_aggeregate=False,
                 attr_recognition=False,
                ):
    device = "cuda"
    if iflog:
        logger = logging.getLogger("reid.test")
        logger.info("Enter inferencing")

    log_path = cfg.LOG_ROOT + cfg.LOG_NAME
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking = reranking, query = query, gallery = gallery, log_path = log_path, gen_result=gen_result, query_aggregate=query_aggeregate)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    torch.cuda.synchronize()
    t0 = time.time()
    bs = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    
    attributes = []
    attr_classes = []
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']

        # attributes
        attrs = informations['others']
        
        
        for k in attrs.keys():
            attrs[k] = attrs[k].to(device)
            attributes.append(attrs[k])
        with torch.no_grad():
            img = img.to(device)
            feat, attr_scores  = model(img, attr_recognition)
            feat = feat[:, 0]
            for scores in attr_scores:
                class_indices = torch.argmax(scores, dim=1)
                attr_classes.append(class_indices.tolist())
                
            evaluator.update((feat, pid, camids))
            img_path_list.extend(imgpath)

    if attr_recognition:
        # if want to get attribute recognition wrong result, set "gen_attr_result = True"
        accuracy_per_attribute = Attribute_Recognition(cfg,attributes,attr_classes,val_loader,gen_attr_reslut = False)
        table = PrettyTable(["task", "gender", "backpack", "hat", "upper_color", "upper_style","lower_color",'lower_style'])
        formatted_accuracy = ["{:.3f}".format(accuracy) for accuracy in accuracy_per_attribute]
        table.add_row(["Attribute Recognition"] + formatted_accuracy)
        logger.info('\n' + str(table))

    total_f_time = time.time() - t0
    single_f_time = total_f_time / (len(val_loader) * img.shape[0])
    num_imgs_per_sec = (len(val_loader) * img.shape[0]) / total_f_time
    if iflog:
        logger.info("Total feature time: {:.2f}s".format(total_f_time))
        logger.info("single feature time: {:.5f}s".format(single_f_time))
        logger.info("number of images per sec: {:.2f}img/s".format(num_imgs_per_sec))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    if iflog:
        logger.info("Validation Results ")
        table = PrettyTable(["task", "mAP", "R1", "R5", "R10"])
        table.add_row(['Reid', mAP, cmc[0],cmc[4], cmc[9]])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        logger.info('\n' + str(table))
        logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP


def do_inference_feat_fusion(cfg,
                 model,
                 val_loader,
                 num_query,
                 reranking=False,
                 iflog=True,
                 query=None,
                 gallery=None,
                 gen_result=False,
                ):
    device = "cuda"
    if iflog:
        logger = logging.getLogger("reid.test")
        logger.info("Enter inferencing")

    log_path = cfg.LOG_ROOT + cfg.LOG_NAME
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking = reranking, query = query, gallery = gallery, log_path = log_path, gen_result=False, query_aggregate=cfg.TEST.QUERY_AGGREGATE)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    torch.cuda.synchronize()
    t0 = time.time()

    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        # domains = informations['others']['domains']
        with torch.no_grad():
            img = img.to(device)
            # camids = camids.to(device)
            feat = model(img, attr_recognition=True)
            feat_sync = feat[:, 0]*0.5 + (feat[:, 1] + feat[:, 3])/2*0.5
            # feat_sync = torch.cat([feat[:, 0], feat[:, 1], feat[:, 3]], dim=1)
            # feat_sync = (feat[:, 1] + feat[:, 2] + feat[:, 3])/3
            evaluator.update((feat_sync, pid, camids))
            img_path_list.extend(imgpath)

    total_f_time = time.time() - t0
    single_f_time = total_f_time / (len(val_loader) * img.shape[0])
    num_imgs_per_sec = (len(val_loader) * img.shape[0]) / total_f_time
    if iflog:
        logger.info("Total feature time: {:.2f}s".format(total_f_time))
        logger.info("single feature time: {:.5f}s".format(single_f_time))
        logger.info("number of images per sec: {:.2f}img/s".format(num_imgs_per_sec))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    if iflog:
        logger.info("Validation Results ")
        table = PrettyTable(["task", "mAP", "R1", "R5", "R10"])
        table.add_row(['Reid', mAP, cmc[0],cmc[4], cmc[9]])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        logger.info('\n' + str(table))
        logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP

def do_inference_multi_targets(cfg,
                 model,
                 logger):

    cmc_all, mAP_all = [0 for i in range(50)], 0
    for testname in cfg.DATASETS.TEST:
        cmc_avg, mAP_avg = [0 for i in range(50)], 0
        for split_id in range(10):
            if testname == 'DG_VIPeR':
                split_id = 'split_{}a'.format(split_id+1)
            val_loader, num_query = build_reid_test_loader(cfg, testname, opt=split_id)
            cmc, mAP = do_inference(cfg, model, val_loader, num_query, False)
            cmc_avg += cmc
            mAP_avg += mAP
        cmc_avg /= 10
        mAP_avg /= 10
        cmc_all += cmc_avg
        mAP_all += mAP_avg
        logger.info("===== Avg Results for 10 splits of {} =====".format(testname))
        logger.info("mAP: {:.1%}".format(mAP_avg))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_avg[r - 1]))

    logger.info("===== Mean Results on 4 target datasets =====")
    logger.info("mAP: {:.1%}".format(mAP_all / len(cfg.DATASETS.TEST)))
    for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_all[r - 1] / len(cfg.DATASETS.TEST)))

    return cmc_all, mAP_all