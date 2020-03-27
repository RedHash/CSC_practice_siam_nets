import torch
from pathlib import Path


# additional
SEED = 1234
N_DEVICES = torch.cuda.device_count()
BASE_PATH = Path("./")
DATA_PATH = BASE_PATH / "data"
WEIGHTS_PATH = BASE_PATH / "weights"
BACKBONE_PATH = WEIGHTS_PATH / "backbone"
TB_LOGDIR = BASE_PATH / "log" / "tb_runs"

BATCH_SIZE = 32
EPOCHS = 20
UNFROZE_EPOCH = 10
NUM_WORKERS = 1
WEIGHT_DECAY = 0.0001
BASE_LR = 0.005
BACKBONE_LR = 0.0005
CLS_WEIGHT = 0.45
LOC_WEIGHT = 0.55
GRAD_CLIP = 10.0

VALID_INTERVAL = 1
VALID_DELAY = 5

D_TEMPLATE = 127
D_DETECTION = 255
OUTPUT_SIZE = 25
HARDNEG_PROB = 0.2
IS_NORM = True
BGR = False

POS_NUM = 16
NEG_NUM = 48
HARDNEG_NUM = 16
THRESHOLD_HIGH = 0.6
THRESHOLD_LOW = 0.3
THRESHOLD_HARDNEG = 0.3

TEMPLATE_SCALE = 0.05
TEMPLATE_BASE_A = 0.5
TEMPLATE_SHIFT = 4

DETECTION_SCALE = 0.18
DETECTION_BASE_A = 1.0
DETECTION_SHIFT = 64  # TODO Why 64 and not 100 (25 * 8)?

SCHEDULER_AFTER_WARMUP = 'Cos'  # Exp or Cos
SCHEDULER_GAMMA = 0.95
SCHEDULER_TMAX = 50
SCHEDULER_WARMUP = 5

EVAL_KWARGS = {'dataset_name': 'VOT', 'root_dir': 'data/vot', 'version': 2016, 'download': True, }
TRACK_BASE_SIZE = 8
TRACK_CONTEXT_AMOUNT = 0.5
TRACK_PENALTY_K = 0.04
TRACK_WINDOW_INFLUENCE = 0.44
TRACK_LR = 0.4

ANCHOR_STRIDE = 8
ANCHOR_RATIOS = [0.33, 0.5, 1, 2, 3]
ANCHOR_SCALES = [8]
ANCHOR_NUM = len(ANCHOR_RATIOS) * len(ANCHOR_SCALES)


# Datesets
TRAIN_DATASETS = ('COCO',)
# COCO
COCO_SAMPLE_RANGE = 0

MODEL_NAME = 'resnet50-imagenet'


class ModelHolder:
    def __init__(self, name):
        self.model_name = name

        global IS_NORM, BGR, ANCHOR_NUM

        if self.model_name == 'resnet50-pysot' or 'resnet50-imagenet':
            self.BACKBONE_TYPE = self.model_name
            self.BACKBONE_KWARGS = {}

            self.CHEST_TYPE = 'Identity'
            self.CHEST_KWARGS = {}

            self.NECK_TYPE = 'AdjustAllLayer'
            self.NECK_KWARGS = {'in_channels': [512, 1024, 2048], 'out_channels': [256, 256, 256]}

            self.RPN_TYPE = "MultiRPN"
            self.RPN_KWARGS = {'anchor_num': ANCHOR_NUM, 'in_channels': [256, 256, 256], 'weighted': True}

            if self.model_name == 'resnet50-pysot':
                IS_NORM = False
                BGR = True

        if self.model_name.startswith('efficientnet'):
            self.BACKBONE_TYPE = self.model_name
            self.BACKBONE_KWARGS = {'use_features': [4, 5, 6], 'strides': [1, 2, 1, 1, 1, 1, 1]}

            self.CHEST_TYPE = 'Identity'
            self.CHEST_KWARGS = {}

            self.NECK_TYPE = 'AdjustAllLayer'
            self.NECK_KWARGS = {'in_channels': [112, 192, 320], 'out_channels': [256, 256, 256]}

            self.RPN_TYPE = "MultiRPN"
            self.RPN_KWARGS = {'anchor_num': ANCHOR_NUM, 'in_channels': [256, 256, 256], 'weighted': True}

        if self.model_name.startswith('efficientdet'):
            det2net = {f'efficientdet-d{i}': f'efficientnet-b{i}' for i in range(8)}

            self.BACKBONE_TYPE = det2net[self.model_name]
            self.BACKBONE_KWARGS = {'use_features': [2, 3, 4, 5, 6], 'strides': [1, 2, 2, 2, 2, 2, 2]}

            self.CHEST_TYPE = 'BIFPN'
            self.CHEST_KWARGS = {'in_channels': [40, 80, 112, 192, 320], 'out_channels': 256, 'stack': 3, 'num_outs': 5}

            self.NECK_TYPE = "AdjustSize"
            self.NECK_KWARGS = {'out_size': 16, 'num_layers': 5}

            self.RPN_TYPE = "MultiRPN"
            self.RPN_KWARGS = {'anchor_num': ANCHOR_NUM, 'in_channels': [256, 256, 256, 256, 256], 'weighted': True}

        self.BACKBONE_WEIGHTS = BACKBONE_PATH / self.BACKBONE_TYPE
