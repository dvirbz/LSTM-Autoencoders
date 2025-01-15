import torch
import torch.nn as nn

class LSTM_AE(nn.Module):
    """
    LSTM Autoencoder for sequence data.
    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Default: True.
    Attributes:
        encoder (nn.LSTM): LSTM layer for encoding the input sequence.
        decoder (nn.LSTM): LSTM layer for decoding the encoded sequence.
    Methods:
        forward(x):
            Forward pass through the autoencoder.
            Args:
                x (torch.Tensor): Input sequence of shape (batch, seq_len, input_size).
            Returns:
                torch.Tensor: Reconstructed sequence of shape (batch, seq_len, input_size).
    """
    
    def __init__(self, input_size, hidden_size, bidirectional=True):
        """
        Initializes the LSTM Autoencoder model.
        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of features in the hidden state.
            bidirectional (bool, optional): If True, the encoder LSTM will be bidirectional. Defaults to True.
        """
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.decoder = nn.LSTM(hidden_size * 2 if bidirectional else hidden_size, input_size, batch_first=True)
        
    def forward(self, x):
        """
        Perform a forward pass through the LSTM Autoencoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Reconstructed input tensor of the same shape as the input.
        """

        z, _ = self.encoder(x)
        x_hat, _ = self.decoder(z)
        return x_hat
    
class LSTM_AE_Classifier(nn.Module):
    """
    LSTM Autoencoder Classifier.
    This model consists of an LSTM-based encoder and decoder, along with a fully connected layer for classification.
    Attributes:
        encoder (nn.LSTM): LSTM layer for encoding the input sequence.
        decoder (nn.LSTM): LSTM layer for decoding the encoded sequence.
        fc (nn.Linear): Fully connected layer for classification.
    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        number_classes (int): The number of output classes for classification.
        bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Default: True.
    Methods:
        forward(x):
            Forward pass through the network.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            Returns:
                x_hat (torch.Tensor): Reconstructed input tensor of shape (batch_size, sequence_length, input_size).
                cls (torch.Tensor): Classification output tensor of shape (batch_size, number_classes).
    """

    def __init__(self, input_size, hidden_size, number_classes, bidirectional=True):
        """
        Initializes the LSTM Autoencoder model.
        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of features in the hidden state.
            number_classes (int): The number of output classes.
            bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Default is True.
        """
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.decoder = nn.LSTM(hidden_size * 2 if bidirectional else hidden_size, input_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, number_classes)
        
    def forward(self, x):
        """
        Perform a forward pass through the LSTM autoencoder.
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, input_size).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - x_hat (torch.Tensor): Reconstructed input tensor of shape (seq_len, batch, input_size).
                - cls (torch.Tensor): Output tensor from the fully connected layer, typically used for classification, of shape (batch, num_classes).
        """
        z, (h_n, _) = self.encoder(x)
        x_hat, _ = self.decoder(z)
        cls = self.fc(h_n.squeeze(0))
        return x_hat, cls

class LSTM_AR(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        """
        Initializes the LSTM Autoencoder model.
        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of features in the hidden state.
            bidirectional (bool, optional): If True, the encoder LSTM will be bidirectional. Defaults to True.
        """
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.decoder = nn.LSTM(hidden_size * 2 if bidirectional else hidden_size, input_size, batch_first=True)
        self.regressor = nn.LSTM(hidden_size * 2 if bidirectional else hidden_size, input_size, batch_first=True)

        
    def forward(self, x):
        """
        Perform a forward pass through the LSTM Autoencoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Reconstructed input tensor of the same shape as the input.
        """

        z, _ = self.encoder(x)
        x_hat, _ = self.decoder(z)
        y_hat, _ = self.regressor(z)
        return x_hat, y_hat
    
    def generate(self, x, N):
        input_x = x
        preds = torch.zeros(x.shape[0], N, x.shape[2])
        for i in range(N):
            z, _ = self.encoder(input_x)
            y_hat, _ = self.regressor(z)
            preds[:, i, :] = y_hat[:, -1, :]
            input_x = torch.cat((input_x, y_hat[:, -1, :]), dim=1)

        return preds
                    
