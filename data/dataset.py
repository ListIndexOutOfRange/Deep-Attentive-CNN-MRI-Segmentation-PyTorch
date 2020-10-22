import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_img
import torch
from torch.utils.data.dataset import Dataset
from typing import Tuple, Optional, Callable, NewType


# Type hint
Transform = NewType('Transform', Optional[Callable[[np.ndarray], torch.Tensor]])


class NiftiDataset(Dataset):

    """ A basic Pytorch Dataset to load nifti couples (image, mask). """

    def __init__(self, rootdir: str, shape: Tuple[int]= (256,256,128), 
                 transform: Transform=None) -> None:
        """ Instanciate a dataset.

        Args:
            rootdir (str): Path to the folder containing the images and masks.
            transform (Transform, optional): Pytorch transformations to apply on couple
                                             (image, mask). Defaults to None.
        """
        super().__init__()
        self.rootdir    = rootdir
        self.image_list = sorted(filter(lambda x: 'src'  in x, os.listdir(self.rootdir)))
        self.mask_list  = sorted(filter(lambda x: 'mask' in x, os.listdir(self.rootdir)))
        self.shape      = np.array(shape)
        self.affine     = self._set_affine()
        self.transform  = transform

    def _set_affine(self) -> np.ndarray:
        """ Generates the affine matrix with respect to specified resolution and shape. 

        Returns:
            np.ndarray: A 4x4 matrix.
        """
        new_resolution = [2,]*3
        new_affine = np.zeros((4,4))
        new_affine[:3,:3] = np.diag(new_resolution)
        new_affine[:3, 3] = self.shape*new_resolution/2.*-1
        new_affine[ 3, 3] = 1.
        return new_affine

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """ Normalize pixel values to [0,1].

        Args:
            data (np.ndarray): A 3D array (from a Nifti image).

        Returns:
            np.ndarray: A 3d array normalized.
        """
        data = data.astype(np.float32)
        return (data - np.min(data))/(np.max(data) - np.min(data))

    def resample(self, data: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
        """ Resample a nifti image, ie changes its affine matrix and its shape
            so that it matches self.shape

        Args:
            data (nib.nifti1.Nifti1Image): A 3D Nifti1 image.

        Returns:
            [type]: A resampled 3D Nifti1 image of shape self.shape.
        """
        return resample_img(data, target_affine=self.affine,
                            target_shape=self.shape, interpolation='nearest')


    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Loads, applies transforms and returns a couple (image, mask).

        Args:
            index (int): The dataset index (one index for one couple (image, mask)).

        Returns:
            Tuple[np.ndarray, np.ndarray]: An image and a mask.
        """
        image = nib.load(os.path.join(self.rootdir, self.image_list[index]))
        mask  = nib.load(os.path.join(self.rootdir,  self.mask_list[index]))
        image, mask = self.resample(image), self.resample(mask)
        image_array = self.normalize(image.get_fdata())
        mask_array  = self.normalize(mask.get_fdata()).astype(np.float32)
        if self.transform is not None:
            image_array, mask_array = self.transform(image_array), self.transform(mask_array)
        image_tensor = torch.from_numpy(image_array)
        mask_tensor  = torch.from_numpy(mask_array)
        image_tensor = image_tensor.permute(2,0,1).unsqueeze(0)
        mask_tensor  = mask_tensor.permute(2,0,1).unsqueeze(0)
        return image_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.image_list)