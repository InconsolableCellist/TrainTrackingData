import torch
from torch import nn

class Bottleneck(nn.Module):
    def __init__(self, latent_size=256):
        super().__init__()
        self.lstm = nn.LSTM(8*8, latent_size//4, batch_first=True)
        self.linear = nn.Linear(latent_size//4, latent_size)
        self.act = nn.ReLU()

    def forward(self, x):
        _, x = self.lstm(x)
        x = x[0].permute(1, 0, 2)
        x = x.squeeze(1)
        x = self.linear(x)
        return x