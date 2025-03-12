import torch
import torch.nn as nn
import math
from typing import Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Implements positional encoding as described in "Attention is All You Need".

        Parameters:
            d_model (int): The dimension of the model.
            dropout (float): Dropout rate.
            max_len (int): Maximum length of sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size: int, model_dim: int, num_heads: int, num_layers: int, output_size: int,
                 dropout: float = 0.1):
        """
        Transformer-based model for time series forecasting.

        Parameters:
            input_size (int): Dimension of input features.
            model_dim (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            output_size (int): Dimension of the output.
            dropout (float): Dropout probability.
        """
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Parameters:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        src = self.input_linear(src)  # (batch_size, seq_length, model_dim)
        src = self.positional_encoding(src)
        # Permute for transformer: (seq_length, batch_size, model_dim)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        # Use the last time step's output for forecasting
        output = self.fc_out(output[-1])
        return output


def train_transformer_model(model: TransformerModel, train_loader, val_loader, num_epochs: int = 10,
                            lr: float = 1e-3) -> TransformerModel:
    """
    Train the Transformer model using MSELoss and Adam optimizer.

    Parameters:
        model (TransformerModel): The Transformer model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.

    Returns:
        TransformerModel: The trained model.
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


def forecast_transformer(model: TransformerModel, input_seq: torch.Tensor) -> torch.Tensor:
    """
    Generate forecast from the Transformer model for a given input sequence.

    Parameters:
        model (TransformerModel): The trained Transformer model.
        input_seq (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Forecasted output.
    """
    model.eval()
    with torch.no_grad():
        prediction = model(input_seq)
    return prediction