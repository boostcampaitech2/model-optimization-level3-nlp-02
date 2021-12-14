"""Inverted Residual v3 block.

Reference:
    https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.modules.activations import HardSigmoid, HardSwish
from src.modules.base_generator import GeneratorAbstract
from src.utils.torch_utils import make_divisible
from src.modules.eca import eca_layer


class ECAInvertedResidualv3(nn.Module):
    """Inverted Residual block MobilenetV3.
    Reference:
        https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
    """

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, k_eca, use_hs):
        super().__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # ECA
                eca_layer(hidden_dim, k_eca, stride, for_v3=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # ECA
                eca_layer(hidden_dim, k_eca, stride, for_v3=True),
                
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor):
        """Forward."""
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)



class ECAInvertedResidualv3Generator(GeneratorAbstract):
    """Bottleneck block generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self._get_divisible_channel(self.args[2] * self.width_multiply)

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        """call method.

        InvertedResidualv3 args consists,
        repeat(=n), [kernel, exp_ratio, out, SE, NL, s] //
        note original notation from paper is [exp_size, out, SE, NL, s]
        """
        module = []
        k, t, _, k_eca, hs, s = self.args  # c is equivalent as self.out_channel
        inp, oup = self.in_channel, self.out_channel
        for i in range(repeat):
            stride = s if i == 0 else 1
            exp_size = self._get_divisible_channel(inp * t)
            module.append(
                self.base_module(
                    inp=inp,
                    hidden_dim=exp_size,
                    oup=oup,
                    kernel_size=k,
                    stride=stride,
                    k_eca=k_eca,
                    use_hs=hs,
                )
            )
            inp = oup
        return self._get_module(module)
