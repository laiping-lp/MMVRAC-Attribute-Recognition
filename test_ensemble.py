import os
from config import cfg
import argparse
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from model.backbones.vit_pytorch import attr_vit_base_patch16_224_TransReID, attr_vit_large_patch16_224_TransReID, attr_vit_small_patch16_224_TransReID, vit_large_patch16_224_TransReID
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

    # output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    output_dir = "/home/liyuke/data4/exp/vit_s8_b12_l16_2_6_2_ensemble_remove_low_resoluton"
    logger = setup_logger("reid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    model_vitb = attr_vit_base_patch16_224_TransReID(stride_size=12)
    model_vitl = attr_vit_large_patch16_224_TransReID(stride_size=16)
    model_vits = attr_vit_small_patch16_224_TransReID(stride_size=8)
    model_vitb.load_param("/home/liyuke/data4/exp/attr_vit_b12_rea_256x128_centerLoss_lr1e-2/attr_vit_best.pth")
    model_vitl.load_param("/home/liyuke/data4/exp/attr_vit_l16_rea_256x128_centerLoss_adamw3.5e-5/attr_vit_best.pth")
    model_vits.load_param("/home/liyuke/data4/exp/attr_vit_s16_rea_256x128_centerLoss_lr1e-2/attr_vit_best.pth")
    
    models = {
        "vit_s": model_vits,
        "vit_b": model_vitb,
        "vit_l": model_vitl,
        }

    for testname in cfg.DATASETS.TEST:
        _, _, val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference_ensemble(cfg, models, val_loader, num_query, reranking=cfg.TEST.RE_RANKING, query_aggregate=True,
                              threshold=0)