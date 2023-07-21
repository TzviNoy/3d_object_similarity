"""
train the autoencoder model and save it.
use later to embedd new data and compare
it to the embedded data of the known data.
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tvt
from tqdm import tqdm

from dataset import VoxelDataset, Noise, Permute, Resize, DepthPad, find_max_shape
from model import AutoEncoder3d


if __name__ == '__main__':

    ROOT = r'C:\Users\tnoy\Documents\Database\educatinal'

    train_voxels = []
    for file in os.listdir(os.path.join(ROOT, 'train')):
        if file.endswith('.npy'):
            train_voxels.append(os.path.join(ROOT, 'train', file))

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
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    model = AutoEncoder3d(embedding_size=4096).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

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
                print(f'epoch: {epoch}, loss: {loss.item()}')

        # save the model
        torch.save(model.state_dict(), 'model.pt')
