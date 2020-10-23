""" Adapted from https://github.com/wulalago/DAF3D/blob/master/Utils.py. """

import torch
from torch import nn


# +-----------------------------------------------------------------------------------------------+ #
# |                                            DICE LOSS                                          | #
# +-----------------------------------------------------------------------------------------------+ #

class DiceLoss(nn.Module):
    
    """ Define the dice loss layer. """
    
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor) -> float:
        """ Compute the dice score of a segmentation result.

        Args:
            input (torch.FloatTensor): Segmentation result.
            target (torch.FloatTensor): Segmentation mask (ground truth).

        Returns:
            float: [description]
        """
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
    segmentation = segmentation.flatten()
    segmentation[segmentation  > 0.5] = 1
    segmentation[segmentation <= 0.5] = 0
    ground_truth = ground_truth.flatten()
    ground_truth[ground_truth > 0.5] = 1
    ground_truth[ground_truth <= 0.5] = 0
    same = (segmentation * ground_truth).sum()
    return 2*float(same)/float(ground_truth.sum() + segmentation.sum())




# +-----------------------------------------------------------------------------------------------+ #
# |                             WEIGHTED COMBINATION OF BCE AND DICE LOSS                         | #
# +-----------------------------------------------------------------------------------------------+ #

class WeightedBCEDiceLoss(nn.Module):
    
    def __init__(self):
        super(WeightedBCEDiceLoss, self).__init__()
        self.bce  = torch.nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, outputs: torch.FloatTensor, targets: torch.FloatTensor) -> float:
        """ A weighted linear combination of several losses from the multi head predictions.

        Args:
            outputs (torch.FloatTensor): Predicted segmentation mask.
            targets (torch.FloatTensor): Ground truth segmentation mask.
            Both of shape (batch_size, channels, depth, width, height).

        Returns:
            (float): The final loss.
        """
        outputs_stage_1, outputs_stage_2, output_stage_3 = outputs
        outputs1_4, outputs1_3, outputs1_2, outputs1_1   = outputs_stage_1
        outputs2_4, outputs2_3, outputs2_2, outputs2_1   = outputs_stage_2
        output     = torch.sigmoid(output_stage_3)
        outputs1_1 = torch.sigmoid(outputs1_1)
        outputs1_2 = torch.sigmoid(outputs1_2)
        outputs1_3 = torch.sigmoid(outputs1_3)
        outputs1_4 = torch.sigmoid(outputs1_4)
        outputs2_1 = torch.sigmoid(outputs2_1)
        outputs2_2 = torch.sigmoid(outputs2_2)
        outputs2_3 = torch.sigmoid(outputs2_3)
        outputs2_4 = torch.sigmoid(outputs2_4)
        loss0_bce  = self.bce(output, targets)
        loss1_bce  = self.bce(outputs1_1, targets)
        loss2_bce  = self.bce(outputs1_2, targets)
        loss3_bce  = self.bce(outputs1_3, targets)
        loss4_bce  = self.bce(outputs1_4, targets)
        loss5_bce  = self.bce(outputs1_1, targets)
        loss6_bce  = self.bce(outputs1_2, targets)
        loss7_bce  = self.bce(outputs1_3, targets)
        loss8_bce  = self.bce(outputs1_4, targets)
        loss0_dice = self.dice(output, targets)
        loss1_dice = self.dice(outputs1_1, targets)
        loss2_dice = self.dice(outputs1_2, targets)
        loss3_dice = self.dice(outputs1_3, targets)
        loss4_dice = self.dice(outputs1_4, targets)
        loss5_dice = self.dice(outputs1_1, targets)
        loss6_dice = self.dice(outputs1_2, targets)
        loss7_dice = self.dice(outputs1_3, targets)
        loss8_dice = self.dice(outputs1_4, targets)
        loss = loss0_bce + 0.4 * loss1_bce + 0.5 * loss2_bce + 0.7 * loss3_bce + 0.8 * loss4_bce + \
                0.4 * loss5_bce + 0.5 * loss6_bce + 0.7 * loss7_bce + 0.8 * loss8_bce + \
                loss0_dice + 0.4 * loss1_dice + 0.5 * loss2_dice + 0.7 * loss3_dice + 0.8 * loss4_dice + \
                0.4 * loss5_dice + 0.7 * loss6_dice + 0.8 * loss7_dice + 1 * loss8_dice
        return loss

















