""" Adapted from https://github.com/wulalago/DAF3D/blob/master/ResNeXt3D.py. """

from functools import partial
from typing import NewType, Callable, Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# +-----------------------------------------------------------------------------------------------+ #
# |                                              UTILS                                            | #
# +-----------------------------------------------------------------------------------------------+ #

def conv3x3x3(in_planes: int, out_planes: int, stride:int=1) -> nn.Module:
    """3x3x3 convolution with padding."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(x: torch.FloatTensor, 
                           planes: int, stride: int) -> torch.autograd.Variable:
    """ Average pooling followed by 0 padding. """
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes-out.size(1), 
                             out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    return Variable(torch.cat([out.data, zero_pads], dim=1))


def get_fine_tuning_parameters(model, ft_begin_index):
    """ Extracts network parameters starting at layer indexed by ft_begin_index. """
    if ft_begin_index == 0:
        return model.parameters()
    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')
    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
            else:
                parameters.append({'params': v, 'lr': 0.0})
    return parameters




# +-----------------------------------------------------------------------------------------------+ #
# |                                            BASIC BLOCK                                        | #
# +-----------------------------------------------------------------------------------------------+ #

class ResNeXtBottleneck(nn.Module):


    """ ResNeXt top -> down basic block. Used for the FPN first two stages. """

    expansion = 2
    def __init__(self, inplanes: int, planes: int, cardinality: int, stride: int=1,
                 downsample: Callable=None) -> None:
        super(ResNeXtBottleneck, self).__init__()
        mid_planes  = cardinality * int(planes / 32)
        self.conv1  = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.gn1    = nn.GroupNorm(32, mid_planes)
        self.conv2  = nn.Conv3d(mid_planes, mid_planes,
                                kernel_size=3,stride=stride, padding=1, 
                                groups=cardinality, bias=False)
        self.gn2    = nn.GroupNorm(32, mid_planes)
        self.conv3  = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.gn3    = nn.GroupNorm(32, planes * self.expansion)
        self.relu   = nn.PReLU()
        self.stride = stride
        self.downsample = downsample

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNeXtDilatedBottleneck(nn.Module):


    """ ResNeXt basic dilated block. Used for the FPN last two stages. """

    expansion = 2
    def __init__(self, inplanes: int, planes: int, cardinality: int, stride: int=1,
                 downsample: Callable=None) -> None:
        super(ResNeXtDilatedBottleneck, self).__init__()
        mid_planes  = cardinality * int(planes / 32)
        self.conv1  = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.gn1    = nn.GroupNorm(32, mid_planes)
        self.conv2  = nn.Conv3d(mid_planes, mid_planes,
                                kernel_size=3, stride=stride, padding=2, dilation=2,
                                groups=cardinality, bias=False)
        self.gn2    = nn.GroupNorm(32, mid_planes)
        self.conv3  = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.gn3    = nn.GroupNorm(32, planes * self.expansion)
        self.relu   = nn.PReLU()
        self.stride = stride
        self.downsample = downsample

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out




# +-----------------------------------------------------------------------------------------------+ #
# |                                            NETWORK                                            | #
# +-----------------------------------------------------------------------------------------------+ #

# Type Hint:
ResidualBlock = NewType('ResidualBlock', Union[ResNeXtBottleneck, ResNeXtDilatedBottleneck])

class ResNeXt3DEncoder(nn.Module):

    """ ResNeXt3D. From paper 'Aggregated Residual Transformations for Deep Neural Networks'. 
        4 stages: Two classic residual blocks, and two residual blocks with dilated convolutions.

        This is a classic ResNeXt 3D implementation, minus the last two layers -
        Adaptavive Average Pooling & Fully Connected.
        Hence acts as a CNN encoder.
    """

    def __init__(self, block: ResidualBlock, layers: List[int], 
                 shortcut_type: str='B', cardinality: int=32) -> None:
        self.inplanes = 64
        super(ResNeXt3DEncoder, self).__init__()
        # stem
        self.conv1 = nn.Conv3d(1, 64,
                               kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.gn1 = nn.GroupNorm(32, 64)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        # first two classic residual blocks
        self.layer1 = self._make_layer(block, 128, layers[0],
                                       shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1],
                                       shortcut_type, cardinality, stride=(1, 2, 2))
        # last two dilated residual blocks
        self.layer3 = self._make_layer(ResNeXtDilatedBottleneck,  512, layers[2],
                                       shortcut_type, cardinality, stride=1)
        self.layer4 = self._make_layer(ResNeXtDilatedBottleneck, 1024, layers[3],
                                       shortcut_type, cardinality, stride=1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion, stride=stride)
            elif shortcut_type == 'B':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, 
                              kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(32, planes * block.expansion),
                )
        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))
        return nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor]:
        """ Pass the input tensor through a 3d ResNeXt minus the final classifier. 

        Args:
            x (torch.FloatTensor): 5d tensor: (batch_size, channels, depth, width, height).

        Returns:
            Tuple[torch.FloatTensor]: Outputs of a residual block after each of the four stages.
        """
        stem = self.relu(self.gn1(self.conv1(x)))
        layer1_output = self.layer1(self.maxpool(stem))
        layer2_output = self.layer2(layer1_output)
        layer3_output = self.layer3(layer2_output)
        layer4_output = self.layer4(layer3_output)
        return layer1_output, layer2_output, layer3_output, layer4_output

    @classmethod
    def from_config(cls, block, model_name, **kwargs):
        """ Instanciate a ResNeCt 3D Encoder using with specified
            basic block and architectures. 
        """
        layers_layout = {
            'resnext3d10' : [1,  1,  1, 1],
            'resnext3d18' : [2,  2,  2, 2],
            'resnext3d34' : [3,  4,  6, 3],
            'resnext3d101': [3,  4, 23, 3],
            'resnext3d152': [3,  8, 35, 3],
            'resnext3d200': [3, 24, 36, 3],
        }
        return cls(block, layers_layout[model_name], **kwargs)