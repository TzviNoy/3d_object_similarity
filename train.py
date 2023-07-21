"""
train the autoencoder model and save it. The trained model will be used in find_neighbors.py
The dataset have examples in various shapes, so we will pad the depth dimension to the max depth
of the voxels in the train set.
Height and width will be resized to the max height and width of the voxels in the train set.
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tvt
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import (VoxelDataset, Noise, Permute, Resize, DepthPad,
                     find_max_shape, plot_voxel_2d)
from model import AutoEncoder3d


def plot_loss(losses):
    plt.plot(losses)
    plt.title('loss')
    plt.show()
    plt.pause(5)
    plt.close()


def train(root: str):
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

    noise = tvt.Compose([Noise(1)])
    dataset = VoxelDataset(ROOT, 'train', transforms, noise)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    model = AutoEncoder3d(embedding_size=4096).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    losses = []
    for epoch in range(epochs):
        for i, (noised_data, data) in tqdm(enumerate(dataloader)):

            noised_data = noised_data.to(device).to(torch.float32)
            data = data.to(device).to(torch.float32)

            optimizer.zero_grad()

            output = model(noised_data)

            loss = criterion(output, data)

            loss.backward()

            optimizer.step()

            if i % 10 == 0:
                losses.append(loss.item())
                print(f'epoch: {epoch}, loss: {loss.item()}')
        # save the model
        torch.save(model.state_dict(), 'model.pt')
        plot_loss(losses)


if __name__ == '__main__':

    ROOT = r'C:\Users\tnoy\Documents\Database\educatinal'
    train(ROOT)
