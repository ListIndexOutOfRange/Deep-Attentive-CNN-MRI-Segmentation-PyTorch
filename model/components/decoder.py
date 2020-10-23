""" Adapted from https://github.com/wulalago/DAF3D/blob/master/DAF3D.py. """

import torch
from torch import nn
from typing import Tuple


# +-----------------------------------------------------------------------------------------------+ #
# |                                         ASSP BASE MODULE                                      | #
# +-----------------------------------------------------------------------------------------------+ #

class ASPPModule(nn.Module):

    """ Atrous Spatial Pyramid Pooling. """

    def __init__(self, inplanes: int, planes: int, rate: int) -> None:
        super(ASPPModule, self).__init__()
        rate_list = (1, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(32, planes)
        self._init_weight()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.atrous_convolution(x)
        x = self.group_norm(x)
        return x

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




# +-----------------------------------------------------------------------------------------------+ #
# |                                           ASPP LAYER                                          | #
# +-----------------------------------------------------------------------------------------------+ #

class ASPPDecoder(nn.Module):

    def __init__(self, rates: Tuple[int]) -> None:
        super(ASPPDecoder, self).__init__()
        self.aspp1 = ASPPModule(64, 64, rate=rates[0])
        self.aspp2 = ASPPModule(64, 64, rate=rates[1])
        self.aspp3 = ASPPModule(64, 64, rate=rates[2])
        self.aspp4 = ASPPModule(64, 64, rate=rates[3])
        self.aspp_conv = nn.Conv3d(256, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        aspp = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)
        aspp = self.aspp_gn(self.aspp_conv(aspp))
        return aspp
