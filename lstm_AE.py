import torch.nn as nn

class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.decoder = nn.LSTM(hidden_size * 2 if bidirectional else hidden_size, input_size, batch_first=True)
        
    def forward(self, x):
        z, _ = self.encoder(x)
        x_hat, _ = self.decoder(z)
        return x_hat
