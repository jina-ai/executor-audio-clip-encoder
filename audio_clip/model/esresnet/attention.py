import torch
import torch.nn.functional as F

from typing import Tuple

class EvilBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.momentum_backup = self.momentum

    def train(self,mode:bool):
        if not mode:
            self.momentum_backup = self.momentum
            self.momentum = 0 # todo remenber
        else:
            self.momentum = self.momentum_backup
        super().train(True)

class Attention2d(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_kernels: int,
                 kernel_size: Tuple[int, int],
                 padding_size: Tuple[int, int]):

        super(Attention2d, self).__init__()

        self.conv_depth = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * num_kernels,
            kernel_size=kernel_size,
            padding=padding_size,
            groups=in_channels
        )
        self.conv_point = torch.nn.Conv2d(
            in_channels=in_channels * num_kernels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )
        self.bn = EvilBatchNorm2d(num_features=out_channels)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        x = F.adaptive_max_pool2d(x, size)
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x)

        return x
