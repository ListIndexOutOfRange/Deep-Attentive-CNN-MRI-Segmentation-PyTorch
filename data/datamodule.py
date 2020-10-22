import os
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from data import NiftiDataset
from typing import Tuple, Optional, Callable, NewType


# Type hint
Transform =  NewType('Transform', Optional[Callable[[np.ndarray], torch.Tensor]])


class DataModule(LightningDataModule):

    """ A Lightning Trainer uses a model and a datamodule. Here is defined a datamodule.
        It's basically a wrapper around dataloaders.
    """  
    
    def __init__(self, input_root: str, shape: Tuple[int]=(256,256,128), 
                 train_batch_size: int=64, val_batch_size: int=64, num_workers: int=4) -> None:
        """ Instanciate a Datamodule able to return three Pytorch DataLoaders (train/val/test).

        Args:
            input_root (str): Path to the folder containing the images and masks.
            train_batch_size (int, optional): Training batch size. Defaults to 64.
            val_batch_size (int, optional): Validation batch size. Defaults to 64.
            num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 4.
        """
        super().__init__()
        self.input_root = input_root
        self.shape = shape
        self.train_batch_size = train_batch_size
        self.val_batch_size   = val_batch_size
        self.num_workers      = num_workers
        self.train_transform, self.test_transform = self.init_transforms()

    def init_transforms(self):
        """ To be implemented. """
        #TODO: make transforms that perfom on 3D MRI images & masks.
        return None, None

    def setup(self, stage: str=None) -> None:
        """ Basically nothing more than train/val split.

        Args:
            stage (str, optional): 'fit' or 'test'.
                                   Init two splitted dataset or one full. Defaults to None.
        """
        total_length = len(os.listdir(self.input_root))//2
        train_length = int(0.8*total_length)
        val_length   = int(0.2*total_length)
        if train_length + val_length != total_length: # round error
            val_length += 1
        if stage == 'fit' or stage is None:
            full_set = NiftiDataset(self.input_root, self.shape, transform=self.train_transform)
            self.train_set, self.val_set = random_split(full_set, [train_length, val_length])
        if stage == 'test' or stage is None:
            self.test_set = NiftiDataset(self.input_root, self.shape, transform=self.test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, num_workers=self.num_workers,
                          batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, num_workers=self.num_workers, 
                          batch_size=self.val_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, num_workers=self.num_workers,
                          batch_size=self.val_batch_size, shuffle=False)

    @classmethod
    def from_config(cls, config):
        return cls(config.rootdir,config.shape, config.train_batch_size,
                   config.val_batch_size, config.num_workers)