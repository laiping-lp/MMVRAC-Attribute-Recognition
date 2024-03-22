import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random
import torch
import numpy as np
import argparse

from config import cfg
from processor.uavhuman_processor import attr_vit_do_train_with_amp
from processor.ori_vit_processor_with_amp import ori_vit_do_train_with_amp
from processor.uavhuman_processor_attr_only import only_attribute_recognition_vit_do_train_with_amp
from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader
# from model.make_model_only_attr import make_model
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.build_loss import build_loss
import loss as Patchloss
from model.extract_features import extract_features
import collections
import yaml
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/uavhuman.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            # logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))
    with open(f'{cfg.LOG_ROOT}/{cfg.LOG_NAME}/log.yml', 'w') as file:
        yaml.dump(cfg, file)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 619) ## uavhuman 619
    # build DG train loader
    train_loader, num_domains, num_pids = build_reid_train_loader(cfg, model)
    cfg.defrost()
    cfg.DATASETS.NUM_DOMAINS = num_domains
    cfg.freeze()
    # build DG validate loader
    val_name = cfg.DATASETS.TEST[0]
    _, _, val_loader, num_query = build_reid_test_loader(cfg, val_name)
    num_classes = len(train_loader.dataset.pids)

    loss_func, center_criterion = build_loss(cfg, num_classes=num_classes)

    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in cfg.MODEL.NAME: # trick from moco v3
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad = False
        print("====== freeze patch_embed for stability ======")

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    # ori_vit_do_train_with_amp(
    if cfg.MODEL.NAME == "vit":
        print("==========> vit!")
        ori_vit_do_train_with_amp(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            num_query, args.local_rank,
            num_pids = num_pids,
        )
    elif "attr_vit" == cfg.MODEL.NAME:
        print("==========> attr_vit!")
        # if want to get best attr model ,set attr_recognition = True
        attr_vit_do_train_with_amp(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            num_query, args.local_rank,
            num_pids = num_pids,
            attr_recognition = True,
        )
    elif "only_attribute_recognition" == cfg.MODEL.NAME:
        print("==========> only_attribute_recognition!")
        only_attribute_recognition_vit_do_train_with_amp(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            num_query, args.local_rank,
            num_pids = num_pids,
        )
    else:
        print("==========> resnet!")
        ori_vit_do_train_with_amp(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            num_query, args.local_rank,
            num_pids = num_pids,
        )