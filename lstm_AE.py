import torch.nn as nn
import torch

class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)
        
    def forward(self, x):
        z, _ = self.encoder(x)
        x_hat, _ = self.decoder(z)
        return x_hat
