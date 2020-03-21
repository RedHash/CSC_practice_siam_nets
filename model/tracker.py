import torch
import torch.nn as nn
import numpy as np
from itertools import chain

from model.backbone import get_backbone
from model.chest import get_chest
from model.neck import get_neck
from model.head import get_rpn_head

import config as cfg
from config import ModelHolder


class SiamTracker(nn.Module):
    xf: torch.Tensor
    zf: torch.Tensor
    metrics: np.ndarray = np.zeros(3)

    def __init__(self):
        super().__init__()
        holder = ModelHolder(cfg.MODEL_NAME)

        self.backbone = get_backbone(holder.BACKBONE_TYPE, holder.BACKBONE_KWARGS)
        self.chest = get_chest(holder.CHEST_TYPE, holder.CHEST_KWARGS)
        self.neck = get_neck(holder.NECK_TYPE, holder.NECK_KWARGS)
        self.rpn_head = get_rpn_head(holder.RPN_TYPE, holder.RPN_KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        zf = self.chest(zf)
        zf = self.neck(zf, crop=True)
        self.zf = zf

    def track(self, x):
        self.xf = self.backbone(x)
        self.xf = self.chest(self.xf)
        self.xf = self.neck(self.xf, crop=False)
        cls, loc = self.rpn_head(self.zf, self.xf)
        return cls, loc

    def forward(self, template, detection):
        zf = self.backbone(template)
        xf = self.backbone(detection)
        zf = self.chest(zf)
        xf = self.chest(xf)
        zf = self.neck(zf, crop=True)
        xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)
        return cls, loc

    def load(self, load_filename):
        path = cfg.WEIGHTS_PATH / load_filename
        state = torch.load(path, map_location=lambda storage, location: storage)
        self.backbone.load_state_dict(state['backbone'])
        self.chest.load_state_dict(state['chest'])
        self.neck.load_state_dict(state['neck'])
        self.rpn_head.load_state_dict(state['head'])

    def save(self, save_filename):
        state = {'backbone': self.backbone.state_dict(),
                 'chest': self.chest.state_dict(),
                 'neck': self.neck.state_dict(),
                 'head': self.rpn_head.state_dict()}
        torch.save(state, cfg.WEIGHTS_PATH / save_filename)

    def load_backbone(self, path):
        self.backbone.load_state_dict(torch.load(path), strict=True)

    def unfroze_trainable(self):
        for param in chain(self.neck.parameters(), self.rpn_head.parameters()):
            param.requires_grad_(True)

    def froze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def unfroze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

    def initialize_weights(self):
        # TODO revisit weights initialization
        for module in chain(self.neck.modules(),
                            self.chest.modules(),
                            self.rpn_head.modules()):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @property
    def accuracy(self):
        return self.metrics[0]

    @property
    def robustness(self):
        return self.metrics[1]

    @property
    def speed_fps(self):
        return self.metrics[3]

    @property
    def backbone_trainable(self):
        return [param for param in self.backbone.parameters() if param.requires_grad]

    @property
    def chest_trainable(self):
        return [param for param in self.chest.parameters() if param.requires_grad]

    @property
    def neck_trainable(self):
        return [param for param in self.neck.parameters() if param.requires_grad]

    @property
    def head_trainable(self):
        return [param for param in self.rpn_head.parameters() if param.requires_grad]


if __name__ == "__main__":
    """ quick test """

    model = SiamTracker()
    z = torch.rand(1, 3, 256, 256)
    x = torch.rand(1, 3, 256, 256)

    cls, loc = model(z, x)
    print(f"cls out: {cls.size()}")
    print(f"loc out: {loc.size()}")

    model.initialize_weights()
