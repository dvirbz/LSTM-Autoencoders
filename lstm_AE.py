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
    
class LSTM_AE_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, number_classes, bidirectional=True):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.decoder = nn.LSTM(hidden_size * 2 if bidirectional else hidden_size, input_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, number_classes)
        
    def forward(self, x):
        z, (h_n, _) = self.encoder(x)
        # print(f"{h_n.shape=}, {z.shape=}")
        x_hat, _ = self.decoder(z)
        cls = self.fc(h_n.squeeze(0))
        # print(f"{cls.dtype=}")
        return x_hat, cls
