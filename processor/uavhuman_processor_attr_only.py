import logging
import os
import time
import torch
import torch.nn as nn
from loss.domain_SCT_loss import domain_SCT_loss
from loss.triplet_loss import euclidean_dist, hard_example_mining
from loss.triplet_loss_for_mixup import hard_example_mining_for_mixup
from model.make_model import make_model
# from model.make_model_only_attr import make_model
from processor.inf_processor import do_inference, do_inference_multi_targets,do_inference_only_attr
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

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
    center_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT

    best = 0.0
    best_index = 1
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_attr_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        
        for n_iter, informations in enumerate(train_loader):
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
                attr_scores = model(img)
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
                attr_log_probs = [nn.LogSoftmax(dim=1)(s) for s in attr_scores] # attr
                loss_attr = [-attr_targets[i] * attr_log_probs[i] for i in range(7)]
                loss_attr = sum([l.mean(0).sum() for l in loss_attr])
                
                loss = loss_attr


            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), bs)
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
            if best < sum(accuracy_per_attribute):
                best = sum(accuracy_per_attribute)
                best_index = epoch
                logger.info("=====best epoch: {}=====".format(best_index))
                logger.info("=====best accuracy: {:.2%}=====".format(best))
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(),
                                os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
                else:
                    torch.save(model.state_dict(),
                            os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
        torch.cuda.empty_cache()
    logger.info("=====best epoch: {}=====".format(best_index))
    logger.info("=====best accuracy: {:.2%}=====".format(best))
    # final evaluation
    load_path = os.path.join(log_path, cfg.MODEL.NAME + '_best.pth')
    model.load_param(load_path)
    logger.info('load weights from best.pth')
    if 'DG' in cfg.DATASETS.TEST[0]:
        do_inference_multi_targets(cfg, model, logger)
    else:
        for testname in cfg.DATASETS.TEST:
            _, _, val_loader, num_query = build_reid_test_loader(cfg, testname)
            do_inference_only_attr(cfg, model, val_loader, num_query)