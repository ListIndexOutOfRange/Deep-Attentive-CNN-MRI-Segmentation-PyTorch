""" Adapted from https://github.com/wulalago/DAF3D/blob/master/DAF3D.py. """

from torch import nn
import torch
import torch.nn.functional as F
from .backbone import ResNeXt3DEncoder


class FPNEncoder(nn.Module):
    
    """ Feature Pyramid Network. 
        4 stages of downsampling (2 with classic convs, 2 with dilated convs.)
        4 stages of upsampling with skip connection to allow multi level features captioning. 
    """
    
    def __init__(self, backbone_block, backbone_name):
        super(FPNEncoder, self).__init__()
        self.backbone = ResNeXt3DEncoder.from_config(backbone_block, backbone_name)
        self.down4 = self._single_step_conv_block(2048, 128) 
        self.down3 = self._single_step_conv_block(1024, 128) 
        self.down2 = self._single_step_conv_block( 512, 128) 
        self.down1 = self._single_step_conv_block( 256, 128) 
        self.fuse1 = self._three_steps_conv_block(512)

    @staticmethod
    def _single_step_conv_block(inplanes, outplanes):
        return nn.Sequential(
            nn.Conv3d(inplanes, outplanes, kernel_size=1), nn.GroupNorm(32, outplanes), nn.PReLU())

    @staticmethod
    def _three_steps_conv_block(inplanes):
        return nn.Sequential(
            nn.Conv3d(inplanes, 64, kernel_size=1),            nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(      64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(      64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU())

    def forward(self, x):
        layer1_output, layer2_output, layer3_output, layer4_output = self.backbone(x)
        # Top-down
        down4 = self.down4(layer4_output)
        down3 = torch.add(F.interpolate(down4, size=layer3_output.size()[2:],
                                        mode='trilinear', align_corners=False),
                          self.down3(layer3_output))
        down2 = torch.add(F.interpolate(down3, size=layer2_output.size()[2:],
                                        mode='trilinear', align_corners=False),
                          self.down2(layer2_output))
        down1 = torch.add(F.interpolate(down2, size=layer1_output.size()[2:],
                                        mode='trilinear', align_corners=False),
                          self.down1(layer1_output))
        down4 = F.interpolate(down4, size=layer1_output.size()[2:],
                              mode='trilinear', align_corners=False)
        down3 = F.interpolate(down3, size=layer1_output.size()[2:],
                              mode='trilinear', align_corners=False)
        down2 = F.interpolate(down2, size=layer1_output.size()[2:],
                              mode='trilinear', align_corners=False)
        return down4, down3, down2, down1 
        