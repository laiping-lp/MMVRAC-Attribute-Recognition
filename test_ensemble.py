import os
from config import cfg
import argparse
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from model.backbones.vit_pytorch import vit_large_patch16_224_TransReID
from processor.inf_processor import do_inference, do_inference_ensemble, do_inference_multi_targets
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/test.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model_vitb = make_model(cfg, cfg.MODEL.NAME, 0)
    model_vitl = vit_large_patch16_224_TransReID(stride_size=12)
    if cfg.TEST.WEIGHT:
        model_vitb.load_param(cfg.TEST.WEIGHT)
    else:
        print("==== random param ====")
    model_vitl.load_param("/data3/laiping/exp/uavhuman_humanbench_vit_large_attr/attr_vit_best.pth")
    
    models = [model_vitb, model_vitl]

    for testname in cfg.DATASETS.TEST:
        _, _, val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference_ensemble(cfg, models, val_loader, num_query, reranking=cfg.TEST.RE_RANKING)