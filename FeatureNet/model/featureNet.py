import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self, vit) -> None:
        super().__init__()
        self.vit = vit

        self.fc1 = nn.Linear(40, 16)
        self.fc2 = nn.Linear(16, 1)
        self.tanh = nn.Hardtanh(0,4)
        
        self.out_fc = nn.Linear(2, 1)
        self.out_tanh = nn.Hardtanh(0,4)

    def _forward_impl(self, x: Tensor, xw: Tensor, xf: Tensor):
        # See note [TorchScript super()]
        
        vit_output = self.vit(x, xw)
        
        xf = self.fc1(xf)
        xf = self.fc2(xf)
        xf = self.tanh(xf)

        concat_x = torch.cat([vit_output, xf], dim=-1)

        output = self.out_fc(concat_x)
        
        return output.double()

    def forward(self, x: Tensor, xw: Tensor, xf: Tensor):
        return self._forward_impl(x, xw, xf)

def featureNet(vit_model):
    return FeatureNet(vit_model)
