from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from petroscope.utils.lazy_imports import torch  # noqa

import geoopt.manifolds.stereographic.math as gmath
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import torch.nn.functional as F

nn = torch.nn  # noqa


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvResBlock, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding="same"
        )
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = self.conv0(x)
        branch1 = self.relu(branch1)
        branch1 = self.bn0(branch1)
        branch2 = self.conv1(x)
        branch2 = self.relu(branch2)
        branch2 = self.bn1(branch2)
        branch2 = self.conv2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.bn2(branch2)
        return branch1 + branch2


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvResBlock(in_channels, out_channels)

    def forward(self, x_down, x_concat):
        x = self.upsample(x_down)
        x = torch.cat((x, x_concat), 1)
        x = self.conv(x)
        return x


class ResUNet(nn.Module):

    def __init__(self, n_classes: int, n_layers: int, start_filters: int):
        super(ResUNet, self).__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.start_filters = start_filters
        self.down_blocks = []
        self.upsample_blocks = []
        self.upconv_blocks = []

        self.down1 = ConvResBlock(3, start_filters)
        self.down2 = ConvResBlock(start_filters, start_filters * 2)
        self.down3 = ConvResBlock(start_filters * 2, start_filters * 4)
        self.down4 = ConvResBlock(start_filters * 4, start_filters * 8)

        self.bottleneck = ConvResBlock(start_filters * 8, start_filters * 16)

        self.up1 = UpBlock(start_filters * 16, start_filters * 8)
        self.up2 = UpBlock(start_filters * 8, start_filters * 4)
        self.up3 = UpBlock(start_filters * 4, start_filters * 2)
        self.up4 = UpBlock(start_filters * 2, start_filters)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out = nn.Conv2d(start_filters, n_classes, kernel_size=1)

    def forward(self, x, exp_name = None, image_name = None, build_radius = False):
        # print('Initial: ', x.shape)
        down1 = self.down1(x)
        mp1 = self.max_pool2d(down1)
        down2 = self.down2(mp1)
        mp2 = self.max_pool2d(down2)
        down3 = self.down3(mp2)
        mp3 = self.max_pool2d(down3)
        down4 = self.down4(mp3)
        mp4 = self.max_pool2d(down4)

        bottleneck = self.bottleneck(mp4)
        # print('Bottleneck: ', bottleneck.shape)

        up1 = self.up1(bottleneck, down4)
        # print('Euclidian embeddings 1: ', up1.shape)
        up2 = self.up2(up1, down3)
        # print('Euclidian embeddings 2: ', up2.shape)
        up3 = self.up3(up2, down2)
        # print('Euclidian embeddings 3: ', up3.shape)
        up4 = self.up4(up3, down1)
        # print('Euclidian embeddings 4: ', up4.shape)
        x_emb = up3
        img = None
        if build_radius:
            mapper = HyperMapper()
            x_h = mapper.expmap(x_emb)
            # print('Hyperbolic embeddings: ', x_h.shape)
            x_h_swapped = torch.swapaxes(x_h, 1, 2)
            x_h_swapped = torch.swapaxes(x_h_swapped, 2, 3)
            # print('Hyperbolic embeddings swapped: ', x_h_swapped.shape)
            r_h = mapper.poincare_distance_origin(x_h_swapped)
            # print('Poincare distance: ', r_h.shape)
            img = r_h.cpu().numpy()
            img = np.moveaxis(img, 0, -1)
            img = np.squeeze(img)
            plt.imshow(img)
            path = f"./out/{exp_name}/radius"
            Path(path).mkdir(parents=True, exist_ok=True)
            plt.axis('off')
            plt.savefig(f"{path}/{image_name}.png", bbox_inches='tight')
            np.save(f"{path}/{image_name}_raw.npy", img)
        res = self.out(up4)
        # print(res.shape)
        return res, img


class HyperMapper(object):
    """A class to map between euclidean and hyperbolic space and compute distances."""

    def __init__(self, c=1.) -> None:
        """Initialize the hyperbolic mapper.

        Args:
            c (float, optional): Hyperbolic curvature. Defaults to 1.0
        """
        self.c = c
        self.K = torch.tensor(-self.c, dtype=float)

    def expmap(self, x, dim=-1):
        """Exponential mapping from Euclidean to hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (..., d)
        """
        x_hyp = gmath.expmap0(x.double(), k=self.K, dim=dim)
        x_hyp = gmath.project(x_hyp, k=self.K, dim=dim)
        return x_hyp
    
    def poincare_distance_origin(self, x, dim=-1):
        """Poincare distance between two points in hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        return gmath.dist0(x, k=self.K, dim=dim)