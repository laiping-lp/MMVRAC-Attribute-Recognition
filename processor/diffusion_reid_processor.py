import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from loss.triplet_loss import euclidean_dist
from processor.inf_processor import do_inference, do_inference_multi_targets
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader

def train_diffusion_reid(cfg,
             models,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             lr_scheduler,
             loss_fn,
             num_query, local_rank,
             num_pids=None,):
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
    
    # if device:
    #     for model in models.values():
    #         model.to(local_rank)
    #         if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
    #             print('Using {} GPUs for training'.format(torch.cuda.device_count()))
    #             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    weight_dtype = torch.float16
    img_extractor = models['image_extractor']
    unet = models['unet']
    vae = models['vae']
    noise_scheduler = models['noise_scheduler']

    loss_meter = AverageMeter()
    loss_id_meter = AverageMeter()
    loss_id_distinct_meter = AverageMeter()
    loss_tri_meter = AverageMeter()
    diffusion_loss_meter = AverageMeter()
    loss_sct_meter = AverageMeter()
    loss_center_meter = AverageMeter()
    loss_xded_meter = AverageMeter()
    loss_tri_hard_meter = AverageMeter()
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
        loss_id_meter.reset()
        loss_id_distinct_meter.reset()
        loss_tri_meter.reset()
        diffusion_loss_meter.reset()
        loss_center_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        lr_scheduler.step(epoch)
        img_extractor.train()
        
        for n_iter, informations in enumerate(train_loader):
            img = informations['images'].to(device)
            vid = informations['targets'].to(device)
            cam = informations['camid'].to(device)
            # ipath = informations['img_path']
            domain = informations['others']['domains'].to(device)

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            cam = cam.to(device)
            domain = domain.to(device)

            targets = torch.zeros((bs, classes)).scatter_(1, target.unsqueeze(1).data.cpu(), 1).to(device)
            with amp.autocast(enabled=True):
                # latents
                latents = vae.encode(img.to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                # Gussian random noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device).long()
                # add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # image feature extraction
                score, feat = img_extractor(img, domain)
                # unet noise prediciton
                idxs = torch.cat([torch.randperm(num_ins)+i*num_ins for i in range(bs // num_ins)], dim=0) #### key, written by lyk
                model_pred = unet(noisy_latents, timesteps, feat[idxs].unsqueeze(1)).sample
                ### diffusion loss
                diffusion_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                ### id loss
                log_probs = nn.LogSoftmax(dim=1)(score)
                targets = 0.9 * targets + 0.1 / classes # label smooth
                loss_id = (- targets * log_probs).mean(0).sum()
                # loss_id = torch.tensor(0.0,device=device) ####### for test

                #### id loss for each domain
                loss_id_distinct = torch.tensor(0.0, device=device)
                # for i,s in enumerate(score_):
                #     if s is None: continue
                #     idx = torch.nonzero(t_domains==i).squeeze()
                #     log_probs = nn.LogSoftmax(1)(s)
                #     label = torch.zeros((len(idx), num_pids[i])).scatter_(1, ori_label[idx].unsqueeze(1).data.cpu(), 1).to(device)
                #     label = 0.9 * label + 0.1 / num_pids[i] # label smooth
                #     loss_id_distinct += (- label * log_probs).mean(0).sum()

                #### triplet loss
                # target = targets.max(1)[1] ###### for mixup
                N = feat.shape[0]
                dist_mat = euclidean_dist(feat, feat)
                # target_new = torch.cat([target,-torch.ones([N], dtype=target.dtype, device=device)], dim=0)
                is_pos = target.expand(N, N).eq(target.expand(N, N).t())
                is_neg = target.expand(N, N).ne(target.expand(N, N).t())
                dist_ap, relative_p_inds = torch.max(
                    dist_mat[is_pos].contiguous().view(bs, -1), 1, keepdim=True)
                dist_an, relative_n_inds = torch.min(
                    dist_mat[is_neg].contiguous().view(bs, -1), 1, keepdim=True)
                y = dist_an.new().resize_as_(dist_an).fill_(1)
                loss_tri = nn.SoftMarginLoss()(dist_an - dist_ap, y)
                # loss_tri = torch.tensor(0.0, device=device)

                #### center loss
                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    loss_center = center_criterion(feat, target)
                else:
                    loss_center = torch.tensor(0.0, device=device)

                loss = loss_id + loss_tri + loss_id_distinct\
                    + center_weight * loss_center + diffusion_loss\

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), bs)
            loss_id_meter.update(loss_id.item(), bs)
            loss_id_distinct_meter.update(loss_id_distinct.item(), bs)
            loss_tri_meter.update(loss_tri.item(), bs)
            diffusion_loss_meter.update(diffusion_loss.item(), bs)
            loss_center_meter.update(center_weight*loss_center.item(), bs)
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, id:{:.3f} tri:{:.3f} dif:{:.3f} cen:{:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader),
                loss_meter.avg, loss_id_meter.avg, loss_tri_meter.avg, diffusion_loss_meter.avg,
                loss_center_meter.avg, acc_meter.avg, lr_scheduler._get_lr(epoch)[0]))
                # tbWriter.add_scalar('train/loss', loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                # tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)

        if epoch % eval_period == 0:
            if 'DG' in cfg.DATASETS.TEST[0]:
                cmc, mAP = do_inference_multi_targets(cfg, img_extractor, num_query, logger)
            else:
                cmc, mAP = do_inference(cfg, img_extractor, val_loader, num_query)
            tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
            tbWriter.add_scalar('val/mAP', mAP, epoch)
            torch.cuda.empty_cache()
            if best < mAP + cmc[0]:
                best = mAP + cmc[0]
                best_index = epoch
                logger.info("=====best epoch: {}=====".format(best_index))
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(img_extractor.state_dict(),
                                os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
                else:
                    torch.save(img_extractor.state_dict(),
                            os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
        torch.cuda.empty_cache()

    # save unet
    unet.save_pretrained(os.path.join(log_path, "unet"))
    # final evaluation
    load_path = os.path.join(log_path, cfg.MODEL.NAME + '_best.pth')
    # eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0)
    img_extractor.load_param(load_path)
    logger.info('load weights from best.pth')
    if 'DG' in cfg.DATASETS.TEST[0]:
        do_inference_multi_targets(cfg, img_extractor, logger)
    else:
        for testname in cfg.DATASETS.TEST:
            val_loader, num_query = build_reid_test_loader(cfg, testname)
            do_inference(cfg, img_extractor, val_loader, num_query)