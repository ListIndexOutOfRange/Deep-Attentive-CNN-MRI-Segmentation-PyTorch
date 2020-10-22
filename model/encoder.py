""" Adapted from https://github.com/wulalago/DAF3D/blob/master/BackBone3D.py. """

from torch import nn
from .resnext_3d import ResNeXt3D, ResNeXtBottleneck, ResNeXtDilatedBottleneck
from typing import NewType, Union


# Type Hint:
ResidualBlock = NewType('ResidualBlock', Union[ResNeXtBottleneck, ResNeXtDilatedBottleneck])


class Encoder3D(nn.Module):
    def __init__(self, block: ResidualBlock=ResNeXtBottleneck,
                 model_name: str='resnext3d34', num_classes: int=2) -> None:
        super(Encoder3D, self).__init__()
        net = ResNeXt3D.from_name(block, model_name, num_classes=num_classes)
        modules = list(net.children())
        # the stem contains the first convolution, bn and relu
        self.stem = nn.Sequential(*modules[:3])
        # the layer1 contains the first pooling and the first 3 bottleneck blocks
        self.layer1 = nn.Sequential(*modules[3:5])
        # the layer2 contains the second 4 bottleneck blocks
        self.layer2 = modules[5]
        # the layer3 contains the first dilated bottleneck blocks
        self.layer3 = modules[6]
        # the layer4 contains the final 3 bottle blocks
        self.layer4 = modules[7]
        # the remaining layers are avg-pooling and dense with num classes uints
        # but we don't use the final two layers in this backbone networks

    def forward(self, x):
        stem = self.stem(x)
        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4