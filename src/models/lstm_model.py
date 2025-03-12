import torch
import torch.nn as nn
from typing import Tuple


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.1):
        """
        LSTM-based model for time series forecasting.

        Parameters:
            input_dim (int): Dimension of the input.
            hidden_dim (int): Number of hidden units.
            num_layers (int): Number of LSTM layers.
            output_dim (int): Dimension of the output.
            dropout (float): Dropout probability.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LSTM model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out


def train_lstm_model(model: LSTMModel, train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader, num_epochs: int = 10, lr: float = 1e-3) -> LSTMModel:
    """
    Train the LSTM model using MSELoss and Adam optimizer.

    Parameters:
        model (LSTMModel): The LSTM model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        LSTMModel: The trained model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    return model


def forecast_lstm(model: LSTMModel, input_seq: torch.Tensor) -> torch.Tensor:
    """
    Generate forecast from the LSTM model for the given input sequence.

    Parameters:
        model (LSTMModel): The trained LSTM model.
        input_seq (torch.Tensor): Input sequence tensor.

    Returns:
        torch.Tensor: The forecasted output.
    """
    model.eval()
    with torch.no_grad():
        prediction = model(input_seq)
    return prediction