import logging
import os
import time
import torch
import torch.nn as nn
from loss.domain_SCT_loss import domain_SCT_loss
from loss.triplet_loss import euclidean_dist, hard_example_mining
from loss.triplet_loss_for_mixup import hard_example_mining_for_mixup
from loss.focal_loss import focal_loss
from model.make_model import make_model
from processor.inf_processor import do_inference, do_inference_multi_targets,do_inference_only_attr
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from loss.center_loss import CenterLoss
from loss.center_loss_attr import CenterLossAttr
from loss.L_softmax_loss import LSoftMaxLoss
from loss.arcface import ArcFace
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

def only_attribute_recognition_vit_do_train_with_amp(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             patch_centers = None,
             pc_criterion= None,
             train_dir=None,
             num_pids=None,
             memories=None,
             sour_centers=None,
             attr_recognition=False):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid.train")
    logger.info('start training')
    log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    tb_path = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print("saving tblog to {}".format(tb_path))
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    loss_attr_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512) # altered by lyk
    bs = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    num_ins = cfg.DATALOADER.NUM_INSTANCE
    classes = len(train_loader.dataset.pids)
    center_weight_attr = cfg.LOSS.CENTER_LOSS_WEIGHT
    
    best = [0.0] * 7
    best_index = [1] * 7
    name = ['Gender','Backpack','Hat','UCC','UCS',"LCC",'LCS']
    # train
    
    if_logsoftmax_with_center_loss = cfg.LOSS.LOGSOFTMAX_CENTER_LOSS
    if_logsoftmax_with_center_loss_attr = cfg.LOSS.LOGSOFTMAX_CENTER_LOSS_ATTR
    if_LSoftmax_loss = cfg.LOSS.LSOFTMAX_LOSS
    logger.info(f" if_LSoftmax_loss:{if_LSoftmax_loss}")
    logger.info(f"if_logsoftmax_with_center_loss:{if_logsoftmax_with_center_loss},if_logsoftmax_with_center_loss_attr:{if_logsoftmax_with_center_loss_attr}")

    if if_logsoftmax_with_center_loss_attr:
        center_criterion_attr = CenterLossAttr(num_classes=12,feat_dim=768,use_gpu=True)
    elif if_logsoftmax_with_center_loss:
        center_criterion_attr = CenterLoss(num_classes=12,feat_dim=768,use_gpu=True)
    margin = cfg.LOSS.L_MARGIN
    lsoftmaxloss = LSoftMaxLoss(num_classes=12,margin=margin,scale=768)
    arcface = ArcFace(in_features=64,out_features=768,m= margin)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_attr_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        
        for n_iter, informations in enumerate(train_loader):
            if(n_iter >= 0):
                img = informations['images'].to(device)
                vid = informations['targets'].to(device)
                target_cam = informations['camid'].to(device)
                ori_label = informations['ori_label'].to(device)

                # attributes
                attrs = informations['others']
                attributes = []
                for k in attrs.keys():
                    attrs[k] = attrs[k].to(device)
                    attributes.append(attrs[k])

                optimizer.zero_grad()
                optimizer_center.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                ori_label = ori_label.to(device)

                targets = torch.zeros((bs, classes)).scatter_(1, target.unsqueeze(1).data.cpu(), 1).to(device)
                model.to(device)
                with amp.autocast(enabled=True):
                    loss_tri_hard = torch.tensor(0.,device=device)
                    feat , attr_scores = model(img)
                    #### attr loss
                    attr_targets = [
                        torch.zeros((bs, 2)).to(device).scatter_(1, attributes[0].unsqueeze(1), 1),
                        torch.zeros((bs, 5)).to(device).scatter_(1, attributes[1].unsqueeze(1), 1),
                        torch.zeros((bs, 5)).to(device).scatter_(1, attributes[2].unsqueeze(1), 1),
                        torch.zeros((bs, 12)).to(device).scatter_(1, attributes[3].unsqueeze(1), 1),
                        torch.zeros((bs, 4)).to(device).scatter_(1, attributes[4].unsqueeze(1), 1),
                        torch.zeros((bs, 12)).to(device).scatter_(1, attributes[5].unsqueeze(1), 1),
                        torch.zeros((bs, 4)).to(device).scatter_(1, attributes[6].unsqueeze(1), 1),
                        ]
                    # LOSS function selection
                    if if_logsoftmax_with_center_loss:
                        # print("======> logsoftmax")
                        attr_log_probs = [nn.LogSoftmax(dim=1)(s) for s in attr_scores] # attr
                        loss_center_attr =  center_criterion_attr(feat,attributes[5]) # LCC loss

                        loss_attr = [-attr_targets[i] * attr_log_probs[i] for i in range(7)]
                        loss_attr = sum([l.mean(0).sum() for l in loss_attr])
                        loss_attr += loss_center_attr * center_weight_attr
                    elif if_LSoftmax_loss:
                        # print("=====> L-SoftMax")
                        lsoftmax_loss_attr = lsoftmaxloss(feat,attributes[5],attr_targets[5],distance_scale=3)
                        attr_log_probs = [nn.LogSoftmax(dim=1)(s) for s in attr_scores] # attr
                        loss_attr = [-attr_targets[i] * attr_log_probs[i] for i in range(7)]
                        loss_attr = sum([l.mean(0).sum() for l in loss_attr])
                        loss_attr += lsoftmax_loss_attr
                    else:
                        attr_log_probs = [nn.LogSoftmax(dim=1)(s) for s in attr_scores] # attr
                        loss_attr = [-attr_targets[i] * attr_log_probs[i] for i in range(7)]
                        loss_attr = sum([l.mean(0).sum() for l in loss_attr])


                                
                    loss = loss_attr   


                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), bs)
                # loss_attr_meter.update(loss_attr.item(), bs)
                loss_attr_meter.update(loss_attr.item(), bs)

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, attr:{:.3f}, Base Lr: {:.2e}"
                    .format(epoch, n_iter+1, len(train_loader),
                    loss_meter.avg, loss_attr_meter.avg, scheduler._get_lr(epoch)[0]))
                    tbWriter.add_scalar('train/loss', loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)

        if epoch % eval_period == 0:
            if 'DG' in cfg.DATASETS.TEST[0]:
                _, _ = do_inference_multi_targets(cfg, model, num_query, logger)
            else:
                accuracy_per_attribute = do_inference_only_attr(cfg, model, val_loader, num_query, attr_recognition=True)
            for i in range(7):
                if(best[i] < accuracy_per_attribute[i]):
                    best[i] = accuracy_per_attribute[i]
                    best_index[i] = epoch
                    # if cfg.MODEL.DIST_TRAIN:
                    #     if dist.get_rank() == 0:
                    #         torch.save(model.state_dict(),
                    #                 os.path.join(log_path, name[i] + '_best.pth'))
                    # elif i in cfg.SAVE_MODEL:
                    #     torch.save(model.state_dict(),
                    #             os.path.join(log_path, name[i] + '_best.pth'))
            table = PrettyTable(["task", "gender", "backpack", "hat", "upper_color", "upper_style","lower_color",'lower_style'])
            formatted_accuracy_per_attribute_best = ["{:.2%}".format(accuracy) for accuracy in best]
            table.add_row(["Attribute Recognition"] + formatted_accuracy_per_attribute_best)
            table.add_row(["best epoch"] + best_index)
            logger.info('\n' + str(table))
            logger.info("=====best accuracy: {:.2%}=====".format(sum(best)))
        torch.cuda.empty_cache()
    table = PrettyTable(["task", "gender", "backpack", "hat", "upper_color", "upper_style","lower_color",'lower_style'])
    formatted_accuracy_per_attribute_best = ["{:.2%}".format(accuracy) for accuracy in best]
    table.add_row(["Attribute Recognition"] + formatted_accuracy_per_attribute_best)
    table.add_row(["best epoch"] + best_index)
    logger.info('\n' + str(table))
    logger.info("=====best accuracy: {:.2%}=====".format(sum(best)))
    logger.info(f"=======save path: {log_path} =======")
    