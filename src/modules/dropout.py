from torch import nn as nn

from src.modules.base_generator import GeneratorAbstract


class DropoutGenerator(GeneratorAbstract):
    """Flatten module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self.in_channel

    def __call__(self, repeat: int = 1):
        return self._get_module(nn.Dropout2d(self.args[0]))