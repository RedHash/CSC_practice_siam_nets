import torch
import torch.nn as nn
import torch.nn.functional as F


def xcorr_depthwise(x, kernel):
    """ depthwise cross correlation """
    batch = kernel.size(0)
    channel = kernel.size(1)

    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))

    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))

    return out


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()

        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)

        return out


class DepthwiseRPN(nn.Module):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)

        return cls, loc


class MultiRPN(nn.Module):
    def __init__(self, anchor_num=5, in_channels=(512, 1024, 2048), weighted=True):
        super(MultiRPN, self).__init__()
        self.weighted = weighted

        for i in range(len(in_channels)):
            self.add_module('rpn' + str(i + 2), DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))

        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)), requires_grad=True)
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)), requires_grad=True)

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []

        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs)):
            rpn = getattr(self, 'rpn' + str(idx + 2))
            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(weight.size(0)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)
