# based on: https://medium.com/data-science/cook-your-first-u-net-in-pytorch-b3297a844cf3

import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv3d function are used to extract features from the input patch.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # Shapes are written as depth x height x width x channels.
        # -------
        # input: 36x48x48x1
        self.e11 = nn.Conv3d(1, 64, kernel_size=3, padding=1)   # output: 36x48x48x64
        self.e12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)  # output: 36x48x48x64
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)      # output: 18x24x24x64

        # input: 18x24x24x64
        self.e21 = nn.Conv3d(64, 128, kernel_size=3, padding=1)   # output: 18x24x24x128
        self.e22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)  # output: 18x24x24x128
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)        # output: 9x12x12x128

        # input: 9x12x12x128
        self.e31 = nn.Conv3d(128, 256, kernel_size=3, padding=1)  # output: 9x12x12x256
        self.e32 = nn.Conv3d(256, 256, kernel_size=3, padding=1)  # output: 9x12x12x256
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)        # output: 4x6x6x256

        # input: 4x6x6x256
        self.e41 = nn.Conv3d(256, 512, kernel_size=3, padding=1)  # output: 4x6x6x512
        self.e42 = nn.Conv3d(512, 512, kernel_size=3, padding=1)  # output: 4x6x6x512
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)        # output: 2x3x3x512

        # input: 2x3x3x512
        self.e51 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)   # output: 2x3x3x1024
        self.e52 = nn.Conv3d(1024, 1024, kernel_size=3, padding=1)  # output: 2x3x3x1024


        # Decoder
        # output_padding=(1, 0, 0) is needed on upconv2 to recover the slice lost
        # when MaxPool3d applied floor division on the odd depth (9 -> 4).
        self.upconv1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)  # output: 4x6x6x512
        self.d11 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, output_padding=(1, 0, 0))  # output: 9x12x12x256
        self.d21 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)  # output: 18x24x24x128
        self.d31 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # output: 36x48x48x64
        self.d41 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # Output layer
        # We have one label per patch (no voxel masks), so the 36x48x48x64 feature map is averaged
        # to one vector of 64 values (global average pooling) and a linear layer gives n_class logits.
        self.pool_out = nn.AdaptiveAvgPool3d(1)  # output: 1x1x1x64
        self.outconv = nn.Linear(64, n_class)    # output: n_class (use n_class=1 with BCEWithLogitsLoss)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.pool_out(xd42).flatten(1)  # (batch, 64, 36, 48, 48) -> (batch, 64)
        out = self.outconv(out)               # (batch, 64) -> (batch, n_class)

        return out