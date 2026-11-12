"""U-Net (Ronneberger, Fischer, Brox 2015) in PyTorch.

Faithful to the paper's Fig. 1:
  * unpadded 3x3 convs + ReLU (so the output is smaller than the input),
  * 2x2 max-pool, channels doubled at each down step,
  * 2x2 up-convolution (transposed conv, stride 2) halving channels,
  * copy-and-crop skip connections (centre-crop contracting features, concat),
  * dropout at the end of the contracting path,
  * final 1x1 conv to n_classes,
  * He initialisation with std = sqrt(2/N), N = incoming nodes of a neuron.
With the defaults: 572x572 input -> 388x388 output, 23 conv layers (19 conv + 4 up-conv).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3), nn.ReLU(inplace=True),
    )


def _crop_like(x, ref):
    dh, dw = x.shape[2] - ref.shape[2], x.shape[3] - ref.shape[3]
    return x[:, :, dh // 2: dh // 2 + ref.shape[2], dw // 2: dw // 2 + ref.shape[3]]


def valid_output_size(inp, depth=4):
    """Output size for a given input size; raises if a max-pool would see an odd size."""
    s = inp
    for _ in range(depth):
        s -= 4
        if s % 2:
            raise ValueError(f"input {inp}: odd feature size {s} before 2x2 max-pool")
        s //= 2
    s -= 4
    for _ in range(depth):
        s = s * 2 - 4
    return s


class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=2, base=64, depth=4, dropout=0.5):
        super().__init__()
        self.depth = depth
        chans = [base * 2 ** i for i in range(depth + 1)]  # 64,128,256,512,1024
        self.enc = nn.ModuleList()
        c = in_ch
        for i in range(depth):
            self.enc.append(_block(c, chans[i]))
            c = chans[i]
        self.bottom = _block(chans[depth - 1], chans[depth])
        self.drop = nn.Dropout2d(dropout)
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2))
            self.dec.append(_block(chans[i] * 2, chans[i]))
        self.head = nn.Conv2d(chans[0], n_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / n))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                # stride-2, kernel-2 transposed conv: each output pixel sees in_channels inputs
                nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / m.in_channels))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            skips.append(x)
            if i == self.depth - 1:
                x = self.drop(x)           # dropout at end of contracting path
            x = F.max_pool2d(x, 2)
        x = self.drop(self.bottom(x))
        for up, blk, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            x = torch.cat([_crop_like(skip, x), x], dim=1)
            x = blk(x)
        return self.head(x)


def count_conv_layers(model):
    return sum(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) for m in model.modules())
