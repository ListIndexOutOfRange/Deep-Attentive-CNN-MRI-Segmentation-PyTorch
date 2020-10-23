""" Base Model Class: A Lighning Module
This class implements all the logic code and will be the one to be fit by a Trainer.
"""

import torch
import torch.optim as optim
import pytorch_lightning as pl
from typing import Tuple, Dict
from .components import FPNEncoder, AttentionModule, ASPPDecoder, MultiHeadPrediction
from .loss import dice_ratio, WeightedBCEDiceLoss


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
        See config.py.Model() to see them all.
        """
        super().__init__()
        self.save_hyperparameters()
        self.encoder   = FPNEncoder(self.hparams.backbone_block, self.hparams.backbone_name)
        self.attention = AttentionModule()
        self.decoder   = ASPPDecoder(self.hparams.aspp_rates)
        self.predict   = MultiHeadPrediction()
        self.loss      = WeightedBCEDiceLoss()

    def configure_optimizers(self) -> Dict:
        """ Instanciate an optimizer and a learning rate scheduler to be used during training.

        Returns:
            (Dict): Dict containing the optimizer(s) and learning rate scheduler(s) to be used by
                    a Trainer object using this model. 
                    The 'monitor' key is used by the ReduceLROnPlateau scheduler.                        
        """
        optimizer = optim.SGD(self.parameters(),
                              lr           = self.hparams.lr,
                              momentum     = self.hparams.momentum,
                              nesterov     = self.hparams.nesterov, 
                              weight_decay = self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode     = self.hparams.rop_mode,
                                                         factor   = self.hparams.rop_factor,
                                                         patience = self.hparams.rop_patience,
                                                         verbose  = self.hparams.verbose)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()[2:]
        single_layer_features  = self.encoder(x)
        prediction_stage_1     = self.predict.after_fpn(single_layer_features, size)
        attentive_feature_maps = self.attention(*single_layer_features) 
        prediction_stage_2     = self.predict.after_attention(attentive_feature_maps[1:], size)
        aspp                   = self.decoder(attentive_feature_maps[0])
        prediction_stage_3     = self.predict.after_assp(aspp, size)
        if self.training:
            return prediction_stage_1, prediction_stage_2, prediction_stage_3
        return prediction_stage_3

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.

        Note that the backward pass is handled under the hood by Pytorch Lightning.

        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be segmented, of shape (N,C,D,W,H).
                                  The other being ...
            batch_idx ([type]): Dataset index of the batch. In range (dataset length)/(batch size).

        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from
                  'hooks' methods from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.log('Loss/Train', loss, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.

        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be classified, of shape (N,C,D,W,H).
                                  The other being ...
            batch_idx (int): Dataset index of the batch. In range (dataset length)/(batch size).

        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from 
                  'hooks' methods from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        inputs, targets = batch
        output  = self(inputs)
        predict = torch.sigmoid(output)
        predict = predict.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        loss = dice_ratio(predict, targets)
        self.log('Loss/Validation', loss, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def test_step(self, batch: torch.Tensor, batch_idx) ->  torch.Tensor:
        """ Not implemented. """
        return None

    @classmethod
    def from_config(cls, config):
        return cls(backbone_block       = config.backbone_block,
                   backbone_name        = config.backbone_name,
                   aspp_rates           = config.aspp_rates,
                   num_classes          = config.num_classes,
                   lr                   = config.lr,
                   momentum             = config.momentum,
                   nesterov             = config.nesterov,
                   weight_decay         = config.weight_decay,
                   rop_mode             = config.rop_mode,
                   rop_factor           = config.rop_factor,
                   rop_patience         = config.rop_patience,
                   verbose              = config.verbose,)