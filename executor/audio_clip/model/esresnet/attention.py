import torch
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple


class EvilBatchNorm2d(torch.nn.BatchNorm2d):
    """Disables usage of running statistics when in eval mode.
    This is necessary to maintain the same behavior in train and eval mode; probably due to a mistake during training.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running_var = None
        self.running_mean = None

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] == 1:
            x = x.expand(2, -1, -1, -1)

        return super().forward(x)


class Attention2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int,
        kernel_size: Tuple[int, int],
        padding_size: Tuple[int, int],
    ):

        super(Attention2d, self).__init__()

        self.conv_depth = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * num_kernels,
            kernel_size=kernel_size,
            padding=padding_size,
            groups=in_channels,
        )
        self.conv_point = torch.nn.Conv2d(
            in_channels=in_channels * num_kernels,
            out_channels=out_channels,
            kernel_size=(1, 1),
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
