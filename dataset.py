"""
Dataset class for the voxel data, and transforms for the voxels
to do it complatible with the model.
"""

from typing import Tuple, Union
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt


class VoxelDataset(Dataset):

    """
    Since the voxels are a numpy arrays, they assumed to be in the shape of [height, width, depth].
    The model expects the voxels to have a depth that is a multiple of the chunk size,
    so we will pad the depth dimension to the nearest multiple of the chunk size.
    """

    def __init__(self, root, folder, transform, noise, chunk_size=16):
        self.transform = transform
        self.noise = noise
        self.chunk_size = chunk_size
        self.files = []

        for file in os.listdir(os.path.join(root, folder)):
            if file.endswith('.npy'):
                self.files.append(os.path.join(root, folder, file))

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[index])

        *_, depth = data.shape

        # make sure the data channel is a multiple of the chunk size
        if self.chunk_size > depth:
            data = np.pad(data, ((0, 0), (0, 0), (0, self.chunk_size - depth)),
                          'constant', constant_values=0)
        if self.chunk_size < depth:
            if depth % self.chunk_size != 0:
                depth_pad = self.chunk_size - (depth % self.chunk_size)
                data = np.pad(data, ((0, 0), (0, 0), (0, depth_pad)), 'constant', constant_values=0)

        data = self.transform(data)
        noised_data = self.noise(data.clone())

        return noised_data, data

    def __len__(self):
        return len(self.files)


class VoxelTestDataset(Dataset):

    """
    Test folder have a subfolder for each class.
    We will return the class folder name as the label.
    """

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

        *_, depth = data.shape

        # make sure the data channel is a multiple of the chunk size
        if self.chunk_size > depth:
            data = np.pad(data, ((0, 0), (0, 0), (0, self.chunk_size - depth)),
                          'constant', constant_values=0)
        if self.chunk_size < depth:
            if depth % self.chunk_size != 0:
                depth_pad = self.chunk_size - (depth % self.chunk_size)
                data = np.pad(data, ((0, 0), (0, 0), (0, depth_pad)), 'constant', constant_values=0)

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
    

class DepthPad:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 0, 0, 0, 0, self.max_depth - x.shape[0]), 'constant', 0)


class Normalize:

    """
    Normalize the voxel according to the given mode. Can be one of the following:
        'min_max' - normalize the voxel to be in the range [0, 1]
        'normal' - normalize the voxel to have a mean of 0 and std of 1
        'given' - normalize the voxel with the given mean and std
    """

    def __init__(self, mode: str, mean: float=torch.nan, std: float=torch.nan):
        self.mode = mode
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        if self.mode == 'min_max':
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        elif self.mode == 'normal':
            x = (x - x.mean()) / (x.std() + 1e-8)

        elif self.mode == 'given':
            x = (x - self.mean) / (self.std + 1e-8)

        else:
            raise ValueError(f'Unknown mode: {self.mode}')

        return x


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


def most_interesting_slice(voxel: np.ndarray, return_location: bool) -> Union[np.ndarray, Tuple[np.ndarray, int, int]]:

    """
    Search for a slice with the highest sum of values and return it.
    If return_location is True, return the slice,
    the dimension and the location of the slice.

    Args:
        voxel (np.ndarray): the voxel to plot
        return_location (bool): whether to return the location of the slice

    Returns:
        slice (np.ndarray): the most interesting slice
        dim (int): the dimension with the highest sum of values
        location (int): the location of the slice
    """

    height_sum = voxel.sum(axis=(1, 2))
    width_sum = voxel.sum(axis=(0, 2))
    depth_sum = voxel.sum(axis=(0, 1))

    max_height = np.argmax(height_sum)
    max_width = np.argmax(width_sum)
    max_depth = np.argmax(depth_sum)

    max_dim = np.argmax(np.array([height_sum.max(), width_sum.max(), depth_sum.max()]))
    max_sum = np.array([max_height, max_width, max_depth])[max_dim]
    voxel2plot = voxel.take(max_sum, axis=max_dim)

    if return_location:
        return voxel2plot, max_dim, max_sum
    return voxel2plot


def plot_voxel_2d(voxel: Union[np.ndarray, torch.Tensor], title: str=''):

    """
    imshow 2d slice of the voxel

    Args:
        voxel (np.ndarray): the voxel to plot
        title (str): the title of the plot

    Returns:
        None
    """

    fig = plt.figure(num=title)
    if fig.axes:
        plt.clf()

    plt.imshow(voxel, cmap='gray')
    plt.title(title)
    plt.show()
    plt.pause(0.001)


def plot_voxel_3d(data: np.ndarray, title: str=''):

    """
    Plot the voxel in 3d

    Args:
        data (np.ndarray): the voxel to plot
        title (str): the title of the plot

    Returns:
        None
    """

    fig = plt.figure(num=title)
    ax = fig.add_subplot(111, projection='3d')
    z, x, y = data.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()


if __name__ == '__main__':

    ROOT = r'C:\Users\tnoy\Documents\Database\educatinal'

    train_voxels = []
    for file in os.listdir(os.path.join(ROOT, 'train')):
        if file.endswith('.npy'):
            train_voxels.append(os.path.join(ROOT, 'train', file))

    max_shape, unsquare_voxels = find_max_shape(train_voxels)
    print(f'max shape: {max_shape}')
    print(f'{len(unsquare_voxels)} voxels are not square')

    plot_voxel_2d(np.load(train_voxels[10]), 'original voxel')
