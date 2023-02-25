import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x_ = self.bn(x)
        x_ = self.relu(x_)
        x = self.relu(x)
        return x_ + x


class ThreeConvLayers(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels)
        self.conv3 = ConvLayer(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def stupid_conv():
    return ThreeConvLayers(3, 10)
