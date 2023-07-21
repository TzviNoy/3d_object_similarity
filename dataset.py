# write an autoencode for 3d data

from typing import Tuple
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


class VoxelDataset(Dataset):

    def __init__(self, root, folder, transform, noise, chunk_size=16):
        self.transform = transform
        self.noise = noise
        self.chunk_size = chunk_size

        self.files = [os.path.join(root, folder, f) for f in (os.listdir(os.path.join(root, folder))) if f.endswith('.npy')]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[index])

        height, width, depth = data.shape
        
        # make sure the data channel is a multiple of the chunk size
        if self.chunk_size > depth:
            data = np.pad(data, ((0, 0), (0, 0), (0, self.chunk_size - depth)), 'constant', constant_values=0)
        if self.chunk_size < depth:
            if depth % self.chunk_size != 0:
                data = np.pad(data, ((0, 0), (0, 0), (0, self.chunk_size - (depth % self.chunk_size))), 'constant', constant_values=0)

        data = self.transform(data)
        noised_data = self.noise(data.clone())

        return noised_data, data

    def __len__(self):
        return len(self.files)


class VoxelTestDataset(Dataset):

    def __init__(self, root, folder, transform, chunk_size=16):

        self.transform = transform
        self.chunk_size = chunk_size
        classes = os.listdir(os.path.join(root, folder))
        self.files = []
        for class_folder in classes:
            for file in os.listdir(os.path.join(root, folder, class_folder)):
                if file.endswith('.npy'):
                    self.files.append(os.path.join(root, folder, class_folder, file))

    def __getitem__(self, index):

        data = np.load(self.files[index])

        class_folder = self.files[index].split('\\')[-2]

        height, width, depth = data.shape

        # make sure the data channel is a multiple of the chunk size
        if self.chunk_size > depth:
            data = np.pad(data, ((0, 0), (0, 0), (0, self.chunk_size - depth)), 'constant', constant_values=0)
        if self.chunk_size < depth:
            if depth % self.chunk_size != 0:
                data = np.pad(data, ((0, 0), (0, 0), (0, self.chunk_size - (depth % self.chunk_size))), 'constant', constant_values=0)

        data = self.transform(data)

        return data, class_folder

    def __len__(self):
        return len(self.files)


class Noise:
    '''Add a random gaussian noise to a 3d tensor'''

    def __init__(self, std=1):
        super().__init__()
        self.std = std

    def __call__(self, x):
        return x + np.random.normal(0, self.std, x.shape)


class Permute:
    def __init__(self, permute: tuple):
        self.permute = permute

    def __call__(self, x):
        return x.permute(self.permute)


class Resize:
    def __init__(self, size: tuple):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x.unsqueeze(1), size=self.size, mode='nearest').squeeze(1)
    

class  DepthPad:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 0, 0, 0, 0, self.max_depth - x.shape[0]), 'constant', 0)

        
def find_max_shape(train_voxels: list)->Tuple[int, list]:

    """
    Find the max shape of the voxels in the train set

    Args:
        train_voxels (list): list of paths to the voxels
    
    Returns:
        max_shape (int): the max shape of the voxels
        unsquare_voxels (list): list of paths to the voxels that are not square
    """

    unsquare_voxels = []
    max_shape = 0
    for voxel in train_voxels:
        with open(voxel, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, version)
            max_shape = max(shape[0], max_shape)
            if not(shape[0] == shape[1] == shape[2]):
                unsquare_voxels.append(voxel)
    return max_shape, unsquare_voxels


if __name__ == '__name__':

    ROOT = r'C:\Users\tnoy\Documents\Database\educatinal'
    
    train_voxels = []
    for file in os.listdir(os.path.join(ROOT, 'train')):
        if file.endswith('.npy'):
            train_voxels.append(os.path.join(ROOT, 'train', file))

    max_shape, unsquare_voxels = find_max_shape(train_voxels)
    print(f'max shape: {max_shape}')
    print(f'{len(unsquare_voxels)} voxels are not square')