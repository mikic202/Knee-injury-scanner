"""ResNet3D model architecture for 3D medical image classification."""

from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling imbalanced datasets.
    Zmniejsza wagę easy examples i koncentruje się na hard examples.
    Formula: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights as tensor (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) - model output logits
            targets: (N,) - target class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers: Sequence[int], num_classes: int = 2, in_channels: int = 1, dropout_rate: float = 0.5):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=(1,2,2), padding=(3,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C=1, D, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_resnet3d(num_classes: int = 3, in_channels: int = 1, layers: Sequence[int] = (2, 2, 2, 2),
                 device: Optional[torch.device] = None, dropout_rate: float = 0.5) -> nn.Module:
    """
    Tworzy ResNet3D podobny do ResNet18 (wariant 3D). Nie wspiera 'pretrained'.
    Domyślne num_classes zmienione na 3 (AD, MCI, CN).
    """
    model = ResNet3D(BasicBlock3D, layers, num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate)
    if device is not None:
        model = model.to(device)
    return model


__all__ = [
    'FocalLoss',
    'BasicBlock3D',
    'ResNet3D',
    'get_resnet3d',
    'conv3x3x3',
    'conv1x1x1',
]
