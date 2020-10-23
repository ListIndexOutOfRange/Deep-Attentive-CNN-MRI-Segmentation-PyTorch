""" Adapted from https://github.com/wulalago/DAF3D/blob/master/DAF3D.py. """


import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadPrediction(nn.Module):
    
    def __init__(self):
        super(MultiHeadPrediction, self).__init__()
        self.predict1_4 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_3 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_2 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_1 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict2_4 = nn.Conv3d( 64, 1, kernel_size=1)
        self.predict2_3 = nn.Conv3d( 64, 1, kernel_size=1)
        self.predict2_2 = nn.Conv3d( 64, 1, kernel_size=1)
        self.predict2_1 = nn.Conv3d( 64, 1, kernel_size=1)
        self.predict    = nn.Conv3d( 64, 1, kernel_size=1)

    def after_fpn(self, fpn_encoder_outputs, size):
        down4, down3, down2, down1 = fpn_encoder_outputs
        predict1_4 = self.predict1_4(down4)
        predict1_3 = self.predict1_3(down3)
        predict1_2 = self.predict1_2(down2)
        predict1_1 = self.predict1_1(down1)
        predict1_1 = F.interpolate(predict1_1, size=size, mode='trilinear', align_corners=False)
        predict1_2 = F.interpolate(predict1_2, size=size, mode='trilinear', align_corners=False)
        predict1_3 = F.interpolate(predict1_3, size=size, mode='trilinear', align_corners=False)
        predict1_4 = F.interpolate(predict1_4, size=size, mode='trilinear', align_corners=False)
        return predict1_4, predict1_3, predict1_2, predict1_1

    def after_attention(self, attentive_features_maps, size):
        refine4, refine3, refine2, refine1 = attentive_features_maps
        predict2_4 = self.predict2_4(refine4)
        predict2_3 = self.predict2_3(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict2_1 = self.predict2_1(refine1)
        predict2_1 = F.interpolate(predict2_1, size=size, mode='trilinear', align_corners=False)
        predict2_2 = F.interpolate(predict2_2, size=size, mode='trilinear', align_corners=False)
        predict2_3 = F.interpolate(predict2_3, size=size, mode='trilinear', align_corners=False)
        predict2_4 = F.interpolate(predict2_4, size=size, mode='trilinear', align_corners=False)
        return predict2_4, predict2_3, predict2_2, predict2_1

    def after_assp(self, aspp_output, size):
        predict = self.predict(aspp_output)
        predict = F.interpolate(predict, size=size, mode='trilinear', align_corners=False)
        return predict        
