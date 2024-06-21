# sample code from torchvision
# https://pytorch.org/vision/master/_modules/torchvision/models/resnet.html#resnet18
# TODO: put your model code here

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT
from vit_pytorch.pit import PiT
from pytorch_pretrained_vit import ViT

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_cnn = conv3x3(in_planes = 1, out_planes = 3)
        self.input_bn = nn.BatchNorm2d(3)
        self.input_relu = nn.ReLU(inplace=True)
        self.vit = ViT('B_16_imagenet1k')

        self.output_fc = nn.Linear(2024, 1)
        
        self.wave_cnn1 = conv3x3(in_planes = 4, out_planes = 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        
        self.wave_cnn2 = conv3x3(in_planes = 32, out_planes = 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.wave_cnn3 = conv3x3(in_planes = 64, out_planes = 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2)
        

        #nn.init.kaiming_uniform_(self.input_cnn.weight, mode="fan_out", nonlinearity="relu")
        #nn.init.kaiming_uniform_(self.output_fc.weight, mode="fan_out", nonlinearity="relu")
    def _forward_impl(self, x: Tensor, xw: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.input_cnn(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        x = self.vit(x)
        
        #x = torch.nn.functional.hardtanh(x, 0, 4)

        xw = self.wave_cnn1(xw)
        xw = self.bn1(xw)
        xw = self.relu1(xw)
        xw = self.maxpool1(xw)
        
        xw = self.wave_cnn2(xw)
        xw = self.bn2(xw)
        xw = self.relu2(xw)
        xw = self.maxpool2(xw)
        
        xw = self.wave_cnn3(xw)
        xw = self.relu3(xw)
        xw = self.maxpool3(xw)
        
        xw = torch.flatten(xw, 1)
        
        total_x = torch.cat([x, xw], dim=1)
        
        total_x = self.output_fc(total_x)

        return total_x

    def forward(self, x: Tensor, xw: Tensor) -> Tensor:
        return self._forward_impl(x, xw)

def ResNet18():
    return ResNet()
