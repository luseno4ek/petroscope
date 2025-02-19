# TODO replace with lazy imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import geoopt.manifolds.stereographic.math as gmath
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import torch.nn.functional as F
import cv2

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=s),
                    nn.Conv2d(
                        in_channels,
                        in_channels // len(pool_sizes),
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels // len(pool_sizes)),
                    nn.ReLU(inplace=True),
                )
                for s in pool_sizes
            ]
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pooled_outputs = [
            F.interpolate(
                feature(x), size=(h, w), mode="bilinear", align_corners=True
            )
            for feature in self.features
        ]
        return torch.cat([x] + pooled_outputs, dim=1)


class PSPNet(nn.Module):
    def __init__(
        self, n_classes: int, backbone: str, dilated=True, pretrained=True
    ):
        super().__init__()

        self.n_classes = n_classes
        self.dilated = dilated
        self.backbone = backbone

        # Load ResNet backbone
        resnet = getattr(models, backbone)(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        if dilated:
            # Apply dilated convolutions to layer3 and layer4
            self.layer3 = self._make_dilated(resnet.layer3, dilation=2)
            self.layer4 = self._make_dilated(resnet.layer4, dilation=4)
        else:
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

        # Determine output channels based on backbone type
        backbone_out_channels = (
            512 if backbone in ["resnet18", "resnet34"] else 2048
        )

        # PSP Module
        self.psp = PyramidPoolingModule(
            in_channels=backbone_out_channels, pool_sizes=(1, 2, 3, 6)
        )

        # Final classifier
        self.final = nn.Sequential(
            nn.Conv2d(
                backbone_out_channels * 2,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, n_classes, kernel_size=1),
        )

    def _make_dilated(self, layer, dilation):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):  # Only modify 3x3 convolutions
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
                if m.stride == (
                    2,
                    2,
                ):  # Prevent downsampling that breaks residuals
                    m.stride = (1, 1)
        return layer

    def forward(self, x, exp_name = None, image_name = None, build_radius = False):
        h, w = x.shape[2], x.shape[3]
        print("Initial shape: ", x.shape)
        x = self.layer0(x)
        print("Layer 0 shape: ", x.shape)
        x = self.layer1(x)
        print("Layer 1 shape: ", x.shape)
        x = self.layer2(x)
        print("Layer 2 shape: ", x.shape)
        x = self.layer3(x)
        print("Layer 3 shape: ", x.shape)
        x = self.layer4(x)
        print("Layer 4 shape: ", x.shape)
        x = self.psp(x)
        print("PSP shape: ", x.shape)
        x = self.final[0](x)
        x = self.final[1](x)
        x = self.final[2](x)
        x = self.final[3](x)
        x_emb = x
        print("Embedding shape: ", x_emb.shape)
        print("Final shape: ", x.shape)
        img = None

        if build_radius:
            mapper = HyperMapper()
            x_h = mapper.expmap(x_emb)
            print('Hyperbolic embeddings: ', x_h.shape)
            x_h_swapped = torch.swapaxes(x_h, 1, 2)
            x_h_swapped = torch.swapaxes(x_h_swapped, 2, 3)
            print('Hyperbolic embeddings swapped: ', x_h_swapped.shape)
            r_h = mapper.poincare_distance_origin(x_h_swapped)
            print('Poincare distance: ', r_h.shape)
            img = r_h.cpu().numpy()
            img = np.moveaxis(img, 0, -1)
            img = np.squeeze(img)
            # img = cv2.resize(img, (w, h), 
            #    interpolation = cv2.INTER_LINEAR)
            plt.imshow(img)
            path = f"./out/{exp_name}/radius"
            Path(path).mkdir(parents=True, exist_ok=True)
            plt.axis('off')
            plt.savefig(f"{path}/{image_name}.png", bbox_inches='tight')
            np.save(f"{path}/{image_name}_raw.npy", img)

        return F.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=True
        ), img


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