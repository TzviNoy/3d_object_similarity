"""
This module contains the model used for training.
"""


from torch import nn
import torch


class AutoEncoder3d(nn.Module):

    """
    As Autoencoder, this model expect to get a noisy input and reconstruct a clean input,
    as close as possible to the original input.

    The model expect a 3d tensor as input, with shape [batch_size, depth, height, width].
    In the forward method the model will add a channel dimension to the input, and permute
    the dimensions to be [batch_size, channel, height, width, depth] as expected by the
    convolutional layers.
    It will divide the input to chunks of size chunk_size and encode each chunk to a
    vector of size embedding_size.
    It will happen through 3d convolutions that will reduce the size of the input in
    the spatial dimensions as well as the depth dimension in factor of 2**4.
    After reducing the size of the input, we will flatten the latent space to be
    [batch, last_conv_channels, flatten_latent_space], where flatten_latent_space equal to:
    [(height / 16) * (width / 16) * (depth / 16)].
    Then we will stack all the chunks together to be [batch_size, num_chunks, flatten_latent_space],
    and embed it to [batch_size, num_chunks, embedding_size] using a linear layer.
    so we have a tensor with shape [batch_size, num_chunks, embedding_size] that will
    be the input to the transformer.
    if return_embedding is True, the model will return the output of the transformer,
    which is a tensor with shape [batch_size, num_chunks, embedding_size].
    if return_embedding is False, the model will return the reconstructed input, which
    is a tensor with shape [batch_size, height, width, depth].

    """

    def __init__(self, chunk_size=16, embedding_size=2048, return_embedding=False):

        super().__init__()
        self.chunk_size = chunk_size
        self.embedding_size = embedding_size
        self.return_embedding = return_embedding

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.linear = nn.Linear(128*16*16, embedding_size)
        self.transformer = nn.Transformer(d_model=embedding_size, nhead=8, num_encoder_layers=6,
                                          num_decoder_layers=6, batch_first=True)
        self.linear2 = nn.Linear(embedding_size, 128*16*16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 2, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 2, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 2, stride=2, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        See class docstring for more information.
        """

        x = x.unsqueeze(1)

        chunks = []
        for i in range(0, x.shape[2], self.chunk_size):
            chunk = x[:, :, i:i + self.chunk_size, :, :].permute(0, 1, 3, 4, 2)
            encoded_chunk = self.encoder(chunk)
            batch, channels, chunk_size, height, width = encoded_chunk.shape
            flattened_chunk = encoded_chunk.flatten(start_dim=1)
            chunks.append(flattened_chunk)

        x = torch.stack(chunks, dim=1)
        x = self.linear(x)

        embedding = self.transformer(x, x)

        if self.return_embedding:
            return embedding.sum(dim=1)

        x = self.linear2(embedding)
        x = x.view(batch, len(chunks), channels, chunk_size, height, width)
    
        decoded_chunks = []
        for i in range(x.shape[1]):
            x_chunk = x[:, i, ...]
            decoded_chunks.append(self.decoder(x_chunk))
        x = torch.cat(decoded_chunks, dim=-1)
        x = x.permute(0, 1, 4, 2, 3).squeeze(1)
        return x


if __name__ == '__main__':
    model = AutoEncoder3d()
    with torch.no_grad():
        inp = torch.randn(1, 64, 256, 256)
        output = model(inp)
    print(output.shape)