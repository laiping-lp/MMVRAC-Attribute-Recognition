from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# META
# -----------------------------------------------------------------------------
# _C.META = CN()

# _C.META.DATA = CN()
# _C.META.DATA.NAMES = ""
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'vit'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# feature dim of the model
_C.MODEL.DIM = 768
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ""

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
#### for clipreid
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

# ID/triplet
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = 16

# patch_embed_type: resnet/ibn/None
_C.MODEL.PATCH_EMBED_TYPE = ''
# fixed patch embed or not
_C.MODEL.FREEZE_PATCH_EMBED = False
# fixed BN or not (CNN)
_C.MODEL.FIXED_RES_BN = False

# local views
_C.MODEL.PC_SCALE = 0.001
_C.MODEL.PC_LOSS = True
_C.MODEL.PC_LR = 0.2
_C.MODEL.PART_NUM = 3

# soft label
_C.MODEL.SOFT_LABEL = True
_C.MODEL.CLUSTER_K = 10 # num of clusters
_C.MODEL.SOFT_WEIGHT = 0.5
_C.MODEL.SOFT_LAMBDA = 0.5

# decoder settings
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.DIM = 512
_C.MODEL.DECODER.DEPTH = 12
_C.MODEL.DECODER.NUM_HEAD = 12
_C.MODEL.DECODER.MLP_RATIO = 4.0
_C.MODEL.DECODER.NORM = 'LN'

# norm settings
_C.MODEL.NORM = CN()
_C.MODEL.NORM.TYPE = 'LN'

# hint & SD settings
_C.MODEL.DISTILL = CN()
_C.MODEL.DISTILL.DO_XDED = False
_C.MODEL.DISTILL.DO_DISTILL = False
_C.MODEL.DISTILL.LAMBDA = 0.2 ## percentage of SD losses.
_C.MODEL.DISTILL.NUM_SELECT_BLOCK = 1
_C.MODEL.DISTILL.START_EPOCH = 1
_C.MODEL.DISTILL.LOSS_TYPE = 'L1'
_C.MODEL.DISTILL.IF_HEAD = False
_C.MODEL.DISTILL.IF_DEEP_SUPERVISE = False

# if only train cls token
_C.MODEL.ONLY_CLS = False

#-----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.MEAN = [123.675, 116.28, 103.53]
# mask like MAE
_C.INPUT.MASK = CN()
_C.INPUT.MASK.ENABLED = False
_C.INPUT.MASK.RATIO = 0.2
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Auto augmentation
_C.INPUT.DO_AUTOAUG = False
# Augmix
_C.INPUT.DO_AUGMIX = False
# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5
# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10
# Random color jitter
_C.INPUT.CJ = CN()
_C.INPUT.CJ.ENABLED = False
_C.INPUT.CJ.PROB = 1.0
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1
# Local Grayscale Transfomation
_C.INPUT.LGT = CN()
_C.INPUT.LGT.DO_LGT = False
_C.INPUT.LGT.PROB = 0.2
# Random Rotation
_C.INPUT.ROTATE = CN()
_C.INPUT.ROTATE.DO_ROTATE = False
_C.INPUT.ROTATE.PROB = 0.5
# Random Patch
_C.INPUT.RPT = CN()
_C.INPUT.RPT.ENABLED = False
_C.INPUT.RPT.PROB = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ('Market1501',)
_C.DATASETS.TEST = ('DukeMTMC',)
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
# combine both train and test sets
_C.DATASETS.COMBINEALL = False
_C.DATASETS.NUM_DOMAINS = 1
_C.DATASETS.TEST_ALL = False


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Naive sampler which don't consider balanced identity sampling
_C.DATALOADER.NAIVE_WAY = True
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
_C.DATALOADER.INDIVIDUAL = False
# camera as domain
_C.DATALOADER.CAMERA_TO_DOMAIN = False # True when single-source
# drop last incomplete batch
_C.DATALOADER.DROP_LAST = False
_C.DATALOADER.DELETE_REM = False # if true, remain idx lower than num_instance
# # random batch or only one domain in a batch
# _C.DATALOADER.RANDOM_BATCH = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005



# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# epoch number of visualization
_C.SOLVER.VIS_PERIOD = 1
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# resume settings
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_PATH = ""

### for clip reid
_C.SOLVER.STAGE1 = CN()
_C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"
_C.SOLVER.STAGE1.BASE_LR = 0.00035
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 0.00001
_C.SOLVER.STAGE1.LR_MIN = 1e-6
_C.SOLVER.STAGE1.WARMUP_METHOD = 'linear'
_C.SOLVER.STAGE1.WEIGHT_DECAY = 1e-4
_C.SOLVER.STAGE1.WEIGHT_DECAY_BIAS = 1e-4
_C.SOLVER.STAGE1.MAX_EPOCHS = 120
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 120
_C.SOLVER.STAGE1.LOG_PERIOD = 50
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5

_C.SOLVER.STAGE2 = CN()
_C.SOLVER.STAGE2.IMS_PER_BATCH= 64
_C.SOLVER.STAGE2.OPTIMIZER_NAME= "Adam"
_C.SOLVER.STAGE2.BASE_LR= 0.000005
_C.SOLVER.STAGE2.WARMUP_METHOD= 'linear'
_C.SOLVER.STAGE2.WARMUP_ITERS= 10
_C.SOLVER.STAGE2.WARMUP_FACTOR=0.1
_C.SOLVER.STAGE2.WEIGHT_DECAY=  0.0001
_C.SOLVER.STAGE2.WEIGHT_DECAY_BIAS= 0.0001
_C.SOLVER.STAGE2.LARGE_FC_LR= False
_C.SOLVER.STAGE2.MAX_EPOCHS= 60
_C.SOLVER.STAGE2.CHECKPOINT_PERIOD= 60
_C.SOLVER.STAGE2.LOG_PERIOD= 60
_C.SOLVER.STAGE2.EVAL_PERIOD= 10
_C.SOLVER.STAGE2.BIAS_LR_FACTOR= 2

_C.SOLVER.STAGE2.STEPS= [30, 50]
_C.SOLVER.STAGE2.GAMMA= 0.1

#### for center loss
_C.SOLVER.STAGE2.CENTER_LR= 0.5
_C.SOLVER.STAGE2.CENTER_LOSS_WEIGHT = 0.0005
# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# If test with query aggregation, options: 'True','False'
_C.TEST.QUERY_AGGREGATE = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = True
# If test attribute recognition options: "True" , "False"
_C.TEST.ATTRIBUTE_RECOGNITION = False

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# root Path to checkpoint and saved log of trained model
_C.LOG_ROOT = ""
# root path to save tensorboard results
_C.TB_LOG_ROOT = ""
# log dir name
_C.LOG_NAME = ""

_C.LOSS = CN()
# extra loss function
_C.LOSS.FOCAL_LOSS = False
_C.LOSS.LOGSOFTMAX_CENTER_LOSS = False
_C.LOSS.LOGSOFTMAX_CENTER_LOSS_ATTR = False
_C.LOSS.ARCFACE = False
_C.LOSS.LSOFTMAX_LOSS = False
_C.LOSS.CENTER_LOSS_WEIGHT = 0.005 ## just for attr center_loss
# L-softmax MARGIN
_C.LOSS.L_MARGIN = 1


_C.SAVE_MODEL = CN()
_C.SAVE_MODEL = []
