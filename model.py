
from torch import nn
import torch


class AutoEncoder(nn.Module):
    """
    get as input a 3d tensor. divide it to chunks of size chunk_size and encode each chunk.
    then sum the encoded chunks and decode the sum.
    use nn.conv2d and nn.convtranspose2d
    use conv3d and convtranspose3d
    """

    def __init__(self, chunk_size=16, return_embedding=False):

        super().__init__()
        self.chunk_size = chunk_size
        self.return_embedding = return_embedding

        self.encoder = nn.Sequential(
            nn.Conv2d(chunk_size, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(128, 128, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, chunk_size, 3, padding=1),
        )

    def forward(self, x):

        chunks = []
        for i in range(0, x.shape[3], self.chunk_size):
            chunk = x[:, i:i + self.chunk_size, :, :]
            chunks.append(self.encoder(chunk))

        x = torch.stack(chunks, dim=2)
        x = torch.sum(x, dim=2).squeeze(2)

        embedding = self.conv(x)
        x = self.decoder(x)

        if self.return_embedding:
            return embedding

        else:
            return x


class AutoEncoder3d(nn.Module):

    """
    This model expect a 3d tensor as input, with shape [batch_size, 1, height, width, depth]
    The encoder will encode each chunk of size chunk_size to a vector of size embedding_size
    using a 3d convolution.
    Then we will treat each chunk as a word in a sentence and use a transformer to encode
    the sentence.
    This will give us the embedding of the input.
    
    The decoder will take the embedding and decode it to the original input.

    The length of the input must be a multiple of chunk_size.
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

    def forward(self, x):

        x = x.unsqueeze(1)
            
        chunks = []
        for i in range(0, x.shape[2], self.chunk_size):
            chunk = x[:, :, i:i + self.chunk_size, :, :].permute(0, 1, 3, 4, 2)
            encoded_chunk = self.encoder(chunk) # [batch_size, 128, chunk_size / 16, height / 16, width / 16]
            batch, channels, chunk_size, height, width = encoded_chunk.shape
            flattened_chunk = encoded_chunk.flatten(start_dim=1) # [batch_size, 128 * chunk_size / 16 * height / 16 * width / 16]
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
        inp = torch.randn(1, 1, 64, 256, 256)
        output = model(inp)
    print(output.shape)