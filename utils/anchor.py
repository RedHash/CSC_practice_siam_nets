import math
import numpy as np
import torch
from itertools import product

from utils.crops import corner2center, center2corner
import config as cfg


class Anchors:

    box_anchors: np.ndarray
    center_anchors: np.ndarray

    def __init__(self, stride, ratios, scales):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.n_anchors = len(self.scales) * len(self.ratios)
        self.base_anchors = np.zeros((self.n_anchors, 4), dtype=np.float32)
        self.generate_base_anchors()

    def generate_base_anchors(self):
        """ creates base_anchors class instance
        which is a 2d numpy array of shape (n_anchors, 4)
        stores (x1, y1, x2, y2) coordinates of each anchor """
        anchor_size = self.stride * self.stride
        combinations = list(product(self.ratios, self.scales))
        for anchor_idx in range(self.n_anchors):
            r, s = combinations[anchor_idx]
            # TODO Why int?
            ws = int(math.sqrt(anchor_size / r))
            w = ws * s
            h = int(ws * r) * s
            self.base_anchors[anchor_idx, :] = \
                np.array([-w * 0.5, -h * 0.5, w * 0.5, h * 0.5])

    def generate_all_anchors(self, img_center, size):
        """ creates box_anchors and center_anchors class instances
        args:
            img_center - center of the search region
            size - size of the output
        box_anchors:
            stores (x1, y1, x2, y2) for each anchor
            Tensor of shape [4, n_anchors, size, size]
        center_anchors:
            (x_center, y_center, width, height) for each anchor
            Tensor of shape [4, n_anchors, size, size]
        """
        a0x = img_center - size // 2 * self.stride
        moved_anchors = self.base_anchors + a0x

        x1 = moved_anchors[:, 0]
        y1 = moved_anchors[:, 1]
        x2 = moved_anchors[:, 2]
        y2 = moved_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.n_anchors, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        x_strides = np.arange(0, size).reshape((1, 1, -1)) * self.stride
        y_strides = np.arange(0, size).reshape((1, -1, 1)) * self.stride

        cx = cx + x_strides
        cy = cy + y_strides

        zero = np.zeros((self.n_anchors, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])

        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.box_anchors = np.stack([x1, y1, x2, y2]).astype(np.float32)
        self.center_anchors = np.stack([cx, cy, w, h]).astype(np.float32)


def get_anchors_numpy():
    anchors = Anchors(cfg.ANCHOR_STRIDE, cfg.ANCHOR_RATIOS, cfg.ANCHOR_SCALES)
    anchors.generate_all_anchors(cfg.D_DETECTION // 2, size=cfg.OUTPUT_SIZE)
    corners, centersizes = anchors.box_anchors, anchors.center_anchors
    corners = corners.transpose([0, 2, 3, 1])
    centersizes = centersizes.transpose([0, 2, 3, 1])
    return corners.reshape([4, -1]).T, centersizes.reshape([4, -1]).T


def get_anchors_torch():
    return [torch.from_numpy(e) for e in get_anchors_numpy()]


if __name__ == '__main__':
    """ quick test """
    pass
