import random
import numpy as np
import math
import torch
from torch.hub import download_url_to_file


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_valid_number(x):
    return not (math.isnan(x) or math.isinf(x) or x > 1e4)


def get_lr(optimizer, group=0):
    return optimizer.param_groups[group]['lr']


def str2maybeNone(v):
    return None if v == 'None' else v


def get_gradnorm(optimizer, group):
    norms = [torch.norm(p.grad).item() for p in optimizer.param_groups[group]['params']]
    return np.mean(norms) if norms else 0


class DummyWriter:
    """ Class for pseudo-tensorboard writer """
    def __init__(self):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def download_backbone_weights(backbone, path):
    backbone_to_url = {
        'resnet18': None,
        'resnet34': None,
        'resnet50-imagenet': 'https://storage.googleapis.com/vot-proj/weights/backbone/resnet50-imagenet',
        'resnet50-pysot': 'https://storage.googleapis.com/vot-proj/weights/backbone/resnet50-pysot',

        'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
        'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
        'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
        'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',
    }
    download_url_to_file(backbone_to_url[backbone], path)


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters
    share common prefix 'module'."""
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path, device):
    pretrained_dict = torch.load(pretrained_path,
                                 map_location=device)

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    model.load_state_dict(pretrained_dict, strict=True)
    return model
