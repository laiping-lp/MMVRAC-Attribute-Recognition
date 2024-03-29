import os
from config import cfg
import argparse
from data.build_DG_dataloader import build_reid_test_loader
# from model.make_model_only_attr import make_model
from model import make_model
from processor.inf_processor import do_inference_only_attr
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
    #         logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0)
    if cfg.TEST.WEIGHT:
        model.load_param(cfg.TEST.WEIGHT)
        logger.info(f"Loading model: =======> {cfg.TEST.WEIGHT}")
    else:
        print("==== random param ====")
    
    weight_name = os.path.basename(cfg.TEST.WEIGHT)
    if "Gender" in weight_name:
        names = ['gender']
    elif 'Backpack' in weight_name:
        names = ['backpack']
    elif "Hat" in weight_name:
        names = ['hat']
    elif "UCC" in weight_name:
        names = ['upper_color']
    elif "UCS" in weight_name:
        names = ['upper_style']
    elif "LCC" in weight_name:
        names = ['lower_color']
    elif "LCS" in weight_name:
        names = ['lower_style']

    for testname in cfg.DATASETS.TEST:
        _, _, val_loader, num_query,_ = build_reid_test_loader(cfg, testname)
        logger.info("=== attribute recognition result ===")
        do_inference_only_attr(cfg, model, val_loader,names=names)