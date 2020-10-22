from dataclasses import dataclass
from typing import NewType, Tuple, Union
from model import ResNeXtBottleneck, ResNeXtDilatedBottleneck

# Type Hint:
ResBlock = NewType('ResBlock', Union[ResNeXtBottleneck, ResNeXtDilatedBottleneck])


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                      DATAMODULE                                     | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

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
    



# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                         TRAIN                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Model:
    
    """ Training config used to instanciate a LightningModel and a Trainer. """

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