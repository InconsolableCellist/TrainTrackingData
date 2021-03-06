import torch
from torch import nn

class Bottleneck(nn.Module):
    def __init__(self, latent_size=256, num_players=20, num_datapoints=42):
        super().__init__()
        # 480 = NUM_PLAYERS (20) * NUM_DATAPOINTS (24)
        # 20 * 42 = 840
        self.lstm = nn.LSTM(num_players * num_datapoints, latent_size//4, batch_first=True)
        self.linear = nn.Linear(latent_size//4, latent_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        #x, player = x
        _, x = self.lstm(x)
        x = x[0].permute(1, 0, 2)
        x = x.squeeze(1)
        x = self.linear(x)
        x = self.dropout(x)
        return x