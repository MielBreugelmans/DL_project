# based on: https://github.com/Tencent/MedicalNet (models/resnet.py)
# Med3D: Transfer Learning for 3D Medical Image Analysis (Chen et al., 2019)
#
# We use ResNet-10, the smallest MedicalNet model:
# - we only have ~1900 training patches, so a deeper network (18/34/50) overfits faster
# - our patches are small (36x48x48), so extra depth adds little receptive field
# - pretrained weights exist for it: resnet_10_23dataset.pth (pretrained on 23 medical datasets)
#
# Changes compared to the original:
# - the segmentation head (conv_seg) is replaced by a classification head: global average pooling + dropout + linear
#   (we have one label per patch)
# - layer names are kept identical, so the pretrained .pth file loads directly

# done with the assistance of Claude AI

import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation,
                     stride=stride, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)

        # shortcut (type B): 1x1 conv when the shape changes, otherwise identity
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes))

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + residual)


class ResNet10(nn.Module):
    # Shapes are written as channels x depth x height x width (our patches: 1x36x48x48)
    def __init__(self, n_class=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # output: 64x18x24x24
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)                 # output: 64x9x12x12

        # ResNet-10 = one BasicBlock per layer (nn.Sequential keeps the names "layer1.0..." of the pretrained file)
        self.layer1 = nn.Sequential(BasicBlock(64, 64))                              # output: 64x9x12x12
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2))                   # output: 128x5x6x6
        self.layer3 = nn.Sequential(BasicBlock(128, 256, dilation=2))                # output: 256x5x6x6 (dilation instead of stride, as in MedicalNet)
        self.layer4 = nn.Sequential(BasicBlock(256, 512, dilation=4))                # output: 512x5x6x6

        # Classification head (replaces conv_seg)
        self.pool_out = nn.AdaptiveAvgPool3d(1)  # output: 512x1x1x1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, n_class)        # output: n_class (use n_class=1 with BCEWithLogitsLoss)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool_out(x).flatten(1)  # (batch, 512, 5, 6, 6) -> (batch, 512)
        return self.fc(self.dropout(x))  # (batch, 512) -> (batch, n_class)
