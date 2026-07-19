"""Lightweight image-to-image refiner + PatchGAN discriminator for UNPAIRED
appearance transfer (content-blind synthetic combos -> real multi-label tile
appearance), used by run_gen_transfer_landcover.py.

Why this design (vs full CycleGAN): the purpose of CycleGAN's cycle-consistency
here is only to preserve the region LAYOUT that defines the multi-hot label while
letting local texture change. We serve that directly and far more stably with a
single residual U-Net generator + an explicit LOW-FREQUENCY content-preservation
loss (L1 on a downsampled image): the coarse layout (labels) is pinned, while the
PatchGAN discriminator -- which judges LOCAL patches, i.e. SPATIAL texture, exactly
the axis CORAL's single global feature-moment could not touch -- pushes the
high-frequency appearance toward the unlabeled real-multi distribution. One
generator + one discriminator is much lighter and more stable in the time budget
than four networks, and LSGAN losses avoid the BCE-GAN saturation instability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _down(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.ReLU(inplace=True),
    )


class UNetRefiner(nn.Module):
    """Residual U-Net: out = clamp(x + delta, 0, 1). The residual form makes the
    identity map trivial to represent, stabilizing early training and biasing the
    generator toward small, texture-level edits of the input (label preservation).
    Input/output are 3-channel images in [0, 1]. Built for 128x128 (3 downsamples).
    """

    def __init__(self, base=32):
        super().__init__()
        self.e0 = nn.Sequential(nn.Conv2d(3, base, 3, padding=1),
                                nn.LeakyReLU(0.2, inplace=True))     # 128
        self.d1 = _down(base, base * 2)                              # 64
        self.d2 = _down(base * 2, base * 4)                         # 32
        self.d3 = _down(base * 4, base * 4)                        # 16
        self.u3 = _up(base * 4, base * 4)                         # 32
        self.u2 = _up(base * 4 * 2, base * 2)                    # 64
        self.u1 = _up(base * 2 * 2, base)                       # 128
        self.out = nn.Conv2d(base * 2, 3, 3, padding=1)

    def forward(self, x):
        e0 = self.e0(x)
        d1 = self.d1(e0)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        u3 = self.u3(d3)
        u2 = self.u2(torch.cat([u3, d2], 1))
        u1 = self.u1(torch.cat([u2, d1], 1))
        delta = self.out(torch.cat([u1, e0], 1))
        return torch.clamp(x + delta, 0.0, 1.0)


class PatchGAN(nn.Module):
    """70x70-ish PatchGAN: outputs a spatial map of real/fake logits so the
    discriminator judges LOCAL texture patches (the spatial appearance signal)."""

    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),                         # 64
            _down(base, base * 2),                                   # 32
            _down(base * 2, base * 4),                              # 16
            nn.Conv2d(base * 4, base * 4, 4, stride=1, padding=1),
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),        # patch logits
        )

    def forward(self, x):
        return self.net(x)


def content_lowfreq(x, factor=4):
    """Downsample-average image -> the coarse layout that must be preserved so the
    multi-hot label stays valid. L1 between content_lowfreq(refined) and
    content_lowfreq(input) is the content-preservation term."""
    return F.avg_pool2d(x, kernel_size=factor, stride=factor)
