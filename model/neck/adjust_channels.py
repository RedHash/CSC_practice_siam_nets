import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjustChannelLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustChannelLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, crop):
        x = self.downsample(x)

        if crop and x.size(3) > 7:
            pad_size = - int((x.size(3) - 7) / 2)
            pad = [pad_size, pad_size, pad_size, pad_size]
            x = F.pad(x, pad)

        return x


class AdjustAllChannelsLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllChannelsLayer, self).__init__()
        self.num = len(in_channels)
        for i in range(self.num):
            self.add_module('downsample' + str(i + 2), AdjustChannelLayer(in_channels[i], out_channels[i]))

    def forward(self, features, crop=False):
        out = []
        for i in range(self.num):
            adj_layer = getattr(self, 'downsample' + str(i + 2))
            out.append(adj_layer(features[i], crop))

        return out


if __name__ == "__main__":
    layer = AdjustChannelLayer(10, 10)

    X = torch.rand(1, 10, 15, 15)
    X = layer(X)

    assert (X.size(3) == 7 and X.size(2) == 7)
