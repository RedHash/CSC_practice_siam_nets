import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjustSizeLayer(nn.Module):
    def __init__(self, out_size):
        super(AdjustSizeLayer, self).__init__()
        self.out_size = out_size

    def forward(self, x, crop):
        x = F.interpolate(x, self.out_size)

        if crop and x.size(3) > 7:
            pad_size = - int((x.size(3) - 7) / 2)
            pad = [pad_size, pad_size, pad_size, pad_size]
            x = F.pad(x, pad)

        return x


class AdjustAllSizesLayer(nn.Module):
    def __init__(self, out_size, num_layers):
        super(AdjustAllSizesLayer, self).__init__()
        self.num = num_layers
        for i in range(self.num):
            self.add_module('downsample' + str(i + 2), AdjustSizeLayer(out_size))

    def forward(self, features, crop=False):
        out = []
        for i in range(self.num):
            adj_layer = getattr(self, 'downsample' + str(i + 2))
            out.append(adj_layer(features[i], crop))
        return out


if __name__ == "__main__":
    new_size = 5
    layer = AdjustSizeLayer(new_size)

    X = torch.rand(1, 1, 3, 3)

    X = layer(X, crop=False)

    assert (X.size(3) == new_size and X.size(2) == new_size)
