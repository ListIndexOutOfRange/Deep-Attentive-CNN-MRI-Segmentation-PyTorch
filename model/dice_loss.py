""" Adapted from https://github.com/wulalago/DAF3D/blob/master/Utils.py. """

import torch
from torch import nn


class DiceLoss(nn.Module):
    
    """ Define the dice loss layer. """
    
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor) -> float:
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def dice_ratio(segmentation: torch.FloatTensor, ground_truth: torch.FloatTensor) -> float:
    """ Define the dice ratio.
    
    Args:
        segmentation (torch.FloatTensor): segmentation result
        ground_truth (torch.FloatTensor): ground truth
    
    Returns:
        (float): Dice ratio of the predictions.
    """
    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0
    gt = gt.flatten()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0
    same = (seg * gt).sum()
    return 2*float(same)/float(gt.sum() + seg.sum())