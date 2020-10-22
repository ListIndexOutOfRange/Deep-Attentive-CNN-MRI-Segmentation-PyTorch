""" Base Model Class: A Lighning Module
This class implements all the logic code and will be the one to be fit by a Trainer.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Dict
from .attentive_cnn import DAF3D
from .dice_loss import DiceLoss, dice_ratio


class LightningModel(pl.LightningModule):
    
    """ LightningModule handling everything training related.
    
    Some attributes and methods used aren't explicitely defined here but comes from the
    LightningModule class. Please refer to the Lightning documentation for further details.

    Note that Lighning Callbacks handles tensorboard logging, early stopping, and auto checkpoints
    for this class. Those are gave to a Trainer object. See init_trainer() in main.py.
    """

    def __init__(self, **kwargs) -> None:
        """ Instanciate a Lightning Model. 
        The call to the Lightning method save_hyperparameters() make every hp accessible through
        self.hparams. e.g: self.hparams.use_targets_smoothing. It also sends them to TensorBoard.
        See model.utils.init_model to see them all.
        """
        super().__init__()
        self.save_hyperparameters()
        self.net  = DAF3D()
        self.bce  = torch.nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def configure_optimizers(self) -> Dict:
        """ Instanciate an optimizer and a learning rate scheduler to be used during training.

        Returns:
            (Dict): Dict containing the optimizer(s) and learning rate scheduler(s) to be used by a Trainer
                    object using this model. 
                    The 'monitor' key is used by the ReduceLROnPlateau scheduler.                        
        """
        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=self.hparams.lr,
                                    momentum=self.hparams.momentum,
                                    nesterov=self.hparams.nesterov, 
                                    weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=self.hparams.rop_mode,
                                                               factor=self.hparams.rop_factor,
                                                               patience=self.hparams.rop_patience,
                                                               verbose=self.hparams.verbose)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.

        Note that the backward pass is handled under the hood by Pytorch Lightning.

        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be segmented, of shape (N,C,D,W,H).
                                  The other being ...
            batch_idx ([type]): Dataset index of the batch. In range (dataset length)/(batch size).

        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from 'hooks' methods
                  from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        inputs, targets = batch
        outputs1, outputs2, outputs3, outputs4, outputs1_1, outputs1_2, outputs1_3, outputs1_4, output = self.net(inputs)
        output = F.sigmoid(output)
        outputs1 = F.sigmoid(outputs1)
        outputs2 = F.sigmoid(outputs2)
        outputs3 = F.sigmoid(outputs3)
        outputs4 = F.sigmoid(outputs4)
        outputs1_1 = F.sigmoid(outputs1_1)
        outputs1_2 = F.sigmoid(outputs1_2)
        outputs1_3 = F.sigmoid(outputs1_3)
        outputs1_4 = F.sigmoid(outputs1_4)
        loss0_bce = self.bce(output, targets)
        loss1_bce = self.bce(outputs1, targets)
        loss2_bce = self.bce(outputs2, targets)
        loss3_bce = self.bce(outputs3, targets)
        loss4_bce = self.bce(outputs4, targets)
        loss5_bce = self.bce(outputs1_1, targets)
        loss6_bce = self.bce(outputs1_2, targets)
        loss7_bce = self.bce(outputs1_3, targets)
        loss8_bce = self.bce(outputs1_4, targets)
        loss0_dice = self.dice(output, targets)
        loss1_dice = self.dice(outputs1, targets)
        loss2_dice = self.dice(outputs2, targets)
        loss3_dice = self.dice(outputs3, targets)
        loss4_dice = self.dice(outputs4, targets)
        loss5_dice = self.dice(outputs1_1, targets)
        loss6_dice = self.dice(outputs1_2, targets)
        loss7_dice = self.dice(outputs1_3, targets)
        loss8_dice = self.dice(outputs1_4, targets)
        loss = loss0_bce + 0.4 * loss1_bce + 0.5 * loss2_bce + 0.7 * loss3_bce + 0.8 * loss4_bce + \
                0.4 * loss5_bce + 0.5 * loss6_bce + 0.7 * loss7_bce + 0.8 * loss8_bce + \
                loss0_dice + 0.4 * loss1_dice + 0.5 * loss2_dice + 0.7 * loss3_dice + 0.8 * loss4_dice + \
                0.4 * loss5_dice + 0.7 * loss6_dice + 0.8 * loss7_dice + 1 * loss8_dice
        self.log('Loss/Train', loss)
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.

        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be classified, of shape (N,C,D,W,H).
                                  The other being ...
            batch_idx (int): Dataset index of the batch. In range (dataset length)/(batch size).

        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from 'hooks' methods
                  from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        inputs, targets = batch
        output  = self.net(inputs)
        predict = F.sigmoid(output)
        predict = predict.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        loss = dice_ratio(predict, targets)
        self.log('Loss/Validation', loss)
        return {'val_loss': loss}

    def test_step(self, batch: torch.Tensor, batch_idx) ->  torch.Tensor:
        """ Not implemented. """

    @classmethod
    def from_config(cls, config):
        return cls(encoder_block        = config.encoder_block,
                   encoder_backbone     = config.encoder_backbone,
                   num_classes          = config.num_classes,
                   lr                   = config.lr,
                   momentum             = config.momentum,
                   nesterov             = config.nesterov,
                   weight_decay         = config.weight_decay,
                   rop_mode             = config.rop_mode,
                   rop_factor           = config.rop_factor,
                   rop_patience         = config.rop_patience,
                   verbose              = config.verbose,)