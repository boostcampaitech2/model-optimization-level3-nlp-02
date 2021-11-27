"""PyTorch Module and ModuleGenerator."""

from src.modules.base_generator import GeneratorAbstract, ModuleGenerator
from src.modules.bottleneck import Bottleneck, BottleneckGenerator
from src.modules.conv import Conv, ConvGenerator, FixedConvGenerator
from src.modules.dwconv import DWConv, DWConvGenerator
from src.modules.fire import Fire, FireGenerator, DropoutGenerator
from src.modules.flatten import FlattenGenerator
from src.modules.invertedresidualv2 import (InvertedResidualv2,
                                            InvertedResidualv2Generator)
from src.modules.invertedresidualv3 import (InvertedResidualv3,
                                            InvertedResidualv3Generator)
from src.modules.linear import Linear, LinearGenerator
from src.modules.poolings import (AvgPoolGenerator, GlobalAvgPool,
                                  GlobalAvgPoolGenerator, MaxPoolGenerator)
from src.modules.eca_invertedresidualv2 import (ECAInvertedResidualv2,
                                                ECAInvertedResidualv2Generator)
from src.modules.eca_invertedresidualv3 import (ECAInvertedResidualv3,
                                                ECAInvertedResidualv3Generator)

__all__ = [
    "ModuleGenerator",
    "GeneratorAbstract",
    "Bottleneck",
    "Conv",
    "DWConv",
    "Linear",
    "GlobalAvgPool",
    "InvertedResidualv2",
    "InvertedResidualv3",
    "Fire",
    "ECAInvertedResidualv2",
    "ECAInvertedResidualv3",
    "BottleneckGenerator",
    "FixedConvGenerator",
    "ConvGenerator",
    "LinearGenerator",
    "DWConvGenerator",
    "FlattenGenerator",
    "MaxPoolGenerator",
    "AvgPoolGenerator",
    "GlobalAvgPoolGenerator",
    "InvertedResidualv2Generator",
    "InvertedResidualv3Generator",
    "FireGenerator",
    "DropoutGenerator",
    "ECAInvertedResidualv2Generator",
    "ECAInvertedResidualv3Generator",
]
