""" Adapted from https://github.com/wulalago/DAF3D/blob/master/DAF3D.py. """

from torch import nn
import torch
import torch.nn.functional as F


class AttentionModule(nn.Module):
    
    def __init__(self):
        super(AttentionModule, self).__init__() 
        self.fuse1 = self._three_steps_conv_block(512)
        self.attention4 = self._attention_block(192)
        self.attention3 = self._attention_block(192)
        self.attention2 = self._attention_block(192)
        self.attention1 = self._attention_block(192)
        self.refine4 = self._three_steps_conv_block(192)
        self.refine3 = self._three_steps_conv_block(192)
        self.refine2 = self._three_steps_conv_block(192)
        self.refine1 = self._three_steps_conv_block(192)
        self.refine = self._single_step_conv_block(256, 64)

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

    @staticmethod
    def _attention_block(inplanes):
        return nn.Sequential(
            nn.Conv3d(inplanes, 64, kernel_size=1),            nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(      64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(      64, 64, kernel_size=3, padding=1), nn.Sigmoid())


    def forward(self, down4, down3, down2, down1):
        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))
        attention4 = self.attention4(torch.cat((down4, fuse1), 1))
        attention3 = self.attention3(torch.cat((down3, fuse1), 1))
        attention2 = self.attention2(torch.cat((down2, fuse1), 1))
        attention1 = self.attention1(torch.cat((down1, fuse1), 1))
        refine4 = self.refine4(torch.cat((down4, attention4 * fuse1), 1))
        refine3 = self.refine3(torch.cat((down3, attention3 * fuse1), 1))
        refine2 = self.refine2(torch.cat((down2, attention2 * fuse1), 1))
        refine1 = self.refine1(torch.cat((down1, attention1 * fuse1), 1))
        refine  = self.refine(torch.cat((refine1, refine2, refine3, refine4), 1))
        return refine, refine4, refine3, refine2, refine1