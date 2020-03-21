import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from utils.anchor import get_anchors_torch


class TrackingLoss(nn.Module):
    """ Implement Loss for Object Tracking """

    def __init__(self, cls_weight, loc_weight):
        super(TrackingLoss, self).__init__()
        assert cls_weight + loc_weight == 1., "Loss sum coefficients should sum into 1. "
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.anchors_corners, self.anchors_centersizes = get_anchors_torch()

    def forward(self, cls_outputs, loc_outputs, gts, pos_anchors, neg_anchors):
        """
        :param cls_outputs: torch.Size([bs, 2 * n_anchs, w, h])
        :param loc_outputs: torch.Size([bs, 4 * n_anchs, w, h])
        :param gts: torch.Size([bs, 4])
        :param pos_anchors: list
        :param neg_anchors: list
        :return:
            loss: torch.Size([])
            values: list
        """
        losses = [self.single_loss(*args) for args
                  in zip(cls_outputs, loc_outputs, pos_anchors, neg_anchors, gts)]

        neg_cls_loss, pos_cls_loss, loc_loss = [self.mean_filter(loss_prt) for loss_prt in zip(*losses)]
        loss = self.cls_weight * (0.5 * neg_cls_loss + 0.5 * pos_cls_loss) + self.loc_weight * loc_loss

        return loss, [self.extract_item(e) for e in (loss, neg_cls_loss, pos_cls_loss, loc_loss)]

    def single_loss(self, cls_out, loc_out, pos_anchs, neg_anchs, gt):

        cls_out = cls_out.view(2, -1).T
        loc_out = loc_out.view(4, -1).T
        neg_cls_loss, pos_cls_loss, loc_loss = 0, 0, 0

        if len(neg_anchs):
            # torch.Size([n_neg_anchors, 2])
            neg_cls_out = cls_out[neg_anchs]
            # torch.Size([n_neg_anchors])
            neg_cls_tgt = torch.zeros(len(neg_anchs), dtype=torch.long, device=cls_out.device)
            # torch.Size([1])
            neg_cls_loss = F.cross_entropy(neg_cls_out, neg_cls_tgt)

        if len(pos_anchs):
            # torch.Size([n_pos_anchors, 2])
            pos_cls_out = cls_out[pos_anchs]
            # torch.Size([n_pos_anchors])
            pos_cls_tgt = torch.ones(len(pos_anchs), dtype=torch.long, device=cls_out.device)
            # torch.Size([1])
            pos_cls_loss = F.cross_entropy(pos_cls_out, pos_cls_tgt)

            # torch.Size([n_pos_anchors, 4])
            loc_out = loc_out[pos_anchs]
            # torch.Size([n_pos_anchors, 4])
            loc_tgt = self.loc_tgt(gt, self.anchors_centersizes[pos_anchs])
            # torch.Size([1])
            loc_loss = F.smooth_l1_loss(loc_out, loc_tgt)

        return neg_cls_loss, pos_cls_loss, loc_loss

    @staticmethod
    def extract_item(x):
        return x.item() if isinstance(x, torch.Tensor) else 0

    @staticmethod
    def mean_filter(x):
        n_nonzero = len(list(filter(lambda e: isinstance(e, torch.Tensor), x)))
        if not n_nonzero:
            return 0.
        return sum(x) / n_nonzero

    @staticmethod
    def loc_tgt(gt, anchs):
        anchs = anchs.to(gt.device)
        delta = gt.unsqueeze(0).expand_as(anchs).clone()
        delta[:, 0:2] -= anchs[:, 0:2]
        delta[:, 0:2] /= anchs[:, 2:4]
        delta[:, 2:4] = torch.log(torch.div(delta[:, 2:4], anchs[:, 2:4]))
        return delta


if __name__ == "__main__":
    """ quick test """

    crt = TrackingLoss(cls_weight=0.5, loc_weight=0.5)
    bs = np.random.randint(2, 10)
    n_anchs, w, h = 5, 25, 25

    cls_out = torch.randn(bs, 2 * n_anchs, w, h)
    loc_out = torch.randn(bs, 4 * n_anchs, w, h)
    gts = torch.randn(bs, 4) ** 2
    pos_anchs = [torch.LongTensor(16).random_(0, n_anchs * w * h) for _ in range(bs)]
    neg_anchs = [torch.LongTensor(48).random_(0, n_anchs * w * h) for _ in range(bs)]

    loss, values = crt(cls_out, loc_out, gts, pos_anchs, neg_anchs)
    assert len(loss.size()) == 0
    assert isinstance(loss, torch.Tensor)
    assert len(values) == 4
