""" This file handles all the hyperparameters of the pipeline.

Those hyperparameters are divided in two dataclasses:
    - DataModule
    - Model
They will be used to instanciate two objects of same names.
Those two objects (DataModule and Model) will be used by a Trainer object
to run the training pipeline.
"""

from dataclasses import dataclass
from typing import NewType, Tuple, Union
from model import ResNeXtBottleneck, ResNeXtDilatedBottleneck

# Type Hint:
ResBlock = NewType('ResBlock', Union[ResNeXtBottleneck, ResNeXtDilatedBottleneck])


# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                          DATAMODULE                                         | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

@dataclass
class DataModule:
    
    """ Preprocessing and data loading config used to instanciate a DataModule object.

    Args:
        input_root (str): The path of the folder containing the images and masks.
        
        shape (Tuple[int]): The shape of the image from MRI to be given to the model. 
        
        train_batch_size (int): Batch size of the training dataloader.

        val_batch_size (int): Batch size of the validation dataloader.

        num_workers (int): Num of threads for the 3 dataloaders (train, val, test).
    """

    rootdir:               str = "/homes/l17vedre/Bureau/Sanssauvegarde/patnum_data/train/"
    shape:          Tuple[int] = (64, 64, 64)
    train_batch_size:      int = 2
    val_batch_size:        int = 2
    num_workers:           int = 4
    



# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                            MODEL                                            | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

@dataclass
class Model:
    
    """ Config to instanciate a LightningModel object. See model/model.py

    Args:
        encoder_block (ResBlock): Which basic residual block is used in the 3D backbone network.

        encoder_backbone (str): Wich 3d backbone network to use. Can be one of:
                                'resnext3d10', 'resnext3d18', 'resnext3d34', 'resnext3d101'
                                'resnext3d152', 'resnext3d200'.

        num_classes (int): Output dim of the final fully connected layer.

        lr (float): Initial learning rate.

        momentum (float): Controls the use of momentum in the optimizer. Float in [0,1].

        nesterov (bool): Controls the use of Nesterov momentum.

        weight_decay: L2 penalty of model's weights. 

        rop_mode: ReduceOnPlateau can monitor a min or a max of a given metrics.

        rop_factor: Learning rate reduction value. new_lr = factor * old_lr.

        rop_patience: How many epochs without metric improving to wait.

        verbose: Should ReduceOnPlateau print when it acts on the learning rate.   
    """

    encoder_block:     ResBlock = ResNeXtBottleneck 
    encoder_backbone:       str = 'resnext3d34'
    num_classes:            int = 2
    lr:                   float = 1e-6
    momentum:             float = 0.9
    nesterov:              bool = True
    weight_decay:         float = 5e-4
    rop_mode:               str = 'min'
    rop_factor:           float = 0.2
    rop_patience:           int = 5
    verbose:               bool = True