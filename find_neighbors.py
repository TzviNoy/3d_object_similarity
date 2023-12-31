"""
Find neighbors of a voxel in the test set using KNN
"""

import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
from torchvision import transforms as tvt
from tqdm import tqdm

from dataset import VoxelTestDataset, Permute, Resize, DepthPad, find_max_shape
from model import AutoEncoder3d


def find_neighbors(root: str):

    """
    We iterate over the test set, encode each voxel and save the embeddings.
    then we use KNN to find the nearest neighbors of each voxel in the test set,
    and compare the predicted class to the true class.

    Args:
        root (str): path to the root folder of the dataset
    """

    train_voxels = []
    for file in os.listdir(os.path.join(root, 'train')):
        if file.endswith('.npy'):
            train_voxels.append(os.path.join(root, 'train', file))

    max_shape, unsquare_voxels = find_max_shape(train_voxels)
    print(f'max shape: {max_shape}')
    print(f'{len(unsquare_voxels)} voxels are not square')

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = tvt.Compose([tvt.ToTensor(), Permute((2, 0, 1)),
                              Resize((max_shape, max_shape)),
                              DepthPad(max_shape)])

    test_set = VoxelTestDataset(root, 'test', transforms)
    test_model = AutoEncoder3d(return_embedding=True).to(device)
    test_model.load_state_dict(torch.load('model.pt'))

    embeddings = []
    obj_cls_list = []
    for i, (data, obj_cls) in enumerate(tqdm(test_set)):
        with torch.no_grad():
            embedding = test_model(data.unsqueeze(0).to(device).to(torch.float32))
        embeddings.append(embedding)
        obj_cls_list.append(obj_cls)

    x = torch.cat(embeddings, dim=0).detach().cpu().numpy()
    y = np.array(obj_cls_list)

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x, y)

    # predict the class of each voxel in the test set
    for i, (data, obj_cls) in enumerate(tqdm(test_set)):
        with torch.no_grad():
            embedding = test_model(data.unsqueeze(0).to(device).to(torch.float32))
        pred = knn.predict(embedding.cpu().numpy())
        print(f'predicted class: {pred}, true class: {obj_cls}')


if __name__ == '__main__':

    ROOT = r'C:\Users\tnoy\Documents\Database\educatinal'
    find_neighbors(ROOT)
