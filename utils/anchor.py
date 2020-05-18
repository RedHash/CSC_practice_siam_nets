import numpy as np
import torch
from itertools import product

import config as cfg


class AnchorsStorage:

    def __init__(self, stride, ratios, scales):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        base_anchors = self.generate_base_anchors(stride, ratios, scales)

        # np.shape([n_anchors, 4])
        centersizes = self.generate_centersizes(base_anchors, stride)
        self.centersizes = centersizes.copy()  # centered
        self.corners = self.generate_corners(centersizes.copy())  # not centered
        self.torch_centersizes = torch.from_numpy(centersizes.copy())  # centered

    @staticmethod
    def generate_base_anchors(stride, ratios, scales):
        """ Func to create base anchors
        :return:
            np.shape([n_base_anchors, 4])
        """
        n_base_anchors = len(scales) * len(ratios)
        base_anchors = np.zeros((n_base_anchors, 4), dtype=np.float32)
        combinations = list(product(ratios, scales))
        for idx in range(n_base_anchors):
            r, s = combinations[idx]
            # TODO int(...) -> cause massive difference, is it correct and why?
            w, h = int(stride * np.sqrt(1/r)) * s, int(stride * np.sqrt(r)) * s
            base_anchors[idx, :] = 0.5 * np.array([-w, -h, w, h])
        return base_anchors

    @staticmethod
    def generate_centersizes(base_anchors, stride):
        """ Func to create centered (cx, cy, w, h) anchors
        :return:
            np.shape([n_anchors, 4])
        """
        score_size = (cfg.D_DETECTION - cfg.D_TEMPLATE) // cfg.ANCHOR_STRIDE + 1 + cfg.TRACK_BASE_SIZE

        x1, y1, x2, y2 = base_anchors.T
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        anchor_num = anchor.shape[0]

        anchor = np.tile(anchor, score_size * score_size).reshape([-1, 4])

        ori = - (score_size // 2) * stride
        xx, yy = np.meshgrid([ori + stride * dx for dx in range(score_size)],
                             [ori + stride * dy for dy in range(score_size)])
        xx, yy = [np.tile(e.flatten(), [anchor_num, 1]).flatten() for e in (xx, yy)]

        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    @staticmethod
    def generate_corners(centersizes):
        """ Func to create (x1, y1, x2, y2) anchors
        :return:
            np.shape([n_anchors, 4])
        """
        centersizes[:, 0:2] += cfg.D_DETECTION // 2
        x, y, w, h = centersizes.T
        return np.stack([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], 1)


Anchors = AnchorsStorage(cfg.ANCHOR_STRIDE, cfg.ANCHOR_RATIOS, cfg.ANCHOR_SCALES)


if __name__ == '__main__':
    """ quick test """

    print(Anchors.centersizes[:10])
    print(Anchors.corners[:10])
