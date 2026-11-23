# based on: https://medium.com/data-science/cook-your-first-u-net-in-pytorch-b3297a844cf3

import torch
import torch.nn as nn
from torch.nn.functional import relu


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # =========================
        # Encoder
        # =========================

        # Input: (batch, 1, 36, 48, 48)
        self.e11 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.bn_e11 = nn.BatchNorm3d(64)

        self.e12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn_e12 = nn.BatchNorm3d(64)

        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Input: (batch, 64, 18, 24, 24)
        self.e21 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn_e21 = nn.BatchNorm3d(128)

        self.e22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn_e22 = nn.BatchNorm3d(128)

        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Input: (batch, 128, 9, 12, 12)
        self.e31 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn_e31 = nn.BatchNorm3d(256)

        self.e32 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn_e32 = nn.BatchNorm3d(256)

        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Input: (batch, 256, 4, 6, 6)
        self.e41 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn_e41 = nn.BatchNorm3d(512)

        self.e42 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn_e42 = nn.BatchNorm3d(512)

        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck: (batch, 512, 2, 3, 3)
        self.e51 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.bn_e51 = nn.BatchNorm3d(1024)

        self.e52 = nn.Conv3d(1024, 1024, kernel_size=3, padding=1)
        self.bn_e52 = nn.BatchNorm3d(1024)

        # =========================
        # Decoder
        # =========================

        self.upconv1 = nn.ConvTranspose3d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        # 512 van upconv + 512 van skip connection = 1024
        self.d11 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.bn_d11 = nn.BatchNorm3d(512)

        self.d12 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn_d12 = nn.BatchNorm3d(512)

        self.upconv2 = nn.ConvTranspose3d(
            512,
            256,
            kernel_size=2,
            stride=2,
            output_padding=(1, 0, 0)
        )

        # 256 van upconv + 256 van skip connection = 512
        self.d21 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.bn_d21 = nn.BatchNorm3d(256)

        self.d22 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn_d22 = nn.BatchNorm3d(256)

        self.upconv3 = nn.ConvTranspose3d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        # 128 van upconv + 128 van skip connection = 256
        self.d31 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.bn_d31 = nn.BatchNorm3d(128)

        self.d32 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn_d32 = nn.BatchNorm3d(128)

        self.upconv4 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        # 64 van upconv + 64 van skip connection = 128
        self.d41 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.bn_d41 = nn.BatchNorm3d(64)

        self.d42 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn_d42 = nn.BatchNorm3d(64)

        # =========================
        # Classification output
        # =========================

        self.pool_out = nn.AdaptiveAvgPool3d(1)
        self.outconv = nn.Linear(64, n_class)

    def forward(self, x):

        # =========================
        # Encoder
        # =========================

        xe11 = relu(self.bn_e11(self.e11(x)))
        xe12 = relu(self.bn_e12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.bn_e21(self.e21(xp1)))
        xe22 = relu(self.bn_e22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.bn_e31(self.e31(xp2)))
        xe32 = relu(self.bn_e32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.bn_e41(self.e41(xp3)))
        xe42 = relu(self.bn_e42(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.bn_e51(self.e51(xp4)))
        xe52 = relu(self.bn_e52(self.e52(xe51)))

        # =========================
        # Decoder
        # =========================

        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)

        xd11 = relu(self.bn_d11(self.d11(xu11)))
        xd12 = relu(self.bn_d12(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)

        xd21 = relu(self.bn_d21(self.d21(xu22)))
        xd22 = relu(self.bn_d22(self.d22(xd21)))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)

        xd31 = relu(self.bn_d31(self.d31(xu33)))
        xd32 = relu(self.bn_d32(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)

        xd41 = relu(self.bn_d41(self.d41(xu44)))
        xd42 = relu(self.bn_d42(self.d42(xd41)))

        # =========================
        # Classification output
        # =========================

        out = self.pool_out(xd42)
        out = out.flatten(1)
        out = self.outconv(out)

        return out