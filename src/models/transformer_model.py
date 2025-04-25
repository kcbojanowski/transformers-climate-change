import torch
import torch.nn as nn
import math
from typing import Tuple


# Custom Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size: int, model_dim: int, num_heads: int, num_layers: int, output_size: int,
                 dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        print(f"Initializing TransformerModel with input_size={input_size}, model_dim={model_dim}, num_heads={num_heads}, num_layers={num_layers}, output_size={output_size}, dropout={dropout}")
        self.input_linear = nn.Linear(input_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                    batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, output_size)
        )

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = self.input_linear(src)
        x = self.positional_encoding(x)

        seq_len = src.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        x = self.transformer_encoder(x, mask=mask)
        x = self.norm(x)
        x = self.fc_out(x[:, -1, :])
        return x

# Darts Transformer Wrapper
try:
    from darts import TimeSeries
    from darts.models import TransformerModel as DartsTransformerModel
except ImportError:
    DartsTransformerModel = None


class DartsTransformerWrapper:
    def __init__(self, config: dict):
        if DartsTransformerModel is None:
            raise ImportError("Darts is not installed. Please install it via 'pip install u8darts'")
        # Filter out parameters that Darts' TransformerModel does not accept
        filtered_config = {k: v for k, v in config.items() if k not in ['lr', 'verbose']}
        self.model = DartsTransformerModel(
            input_chunk_length=filtered_config["input_chunk_length"],
            output_chunk_length=filtered_config["output_chunk_length"],
            d_model=filtered_config["d_model"],
            nhead=filtered_config["nhead"],
            num_encoder_layers=filtered_config["num_encoder_layers"],
            num_decoder_layers=filtered_config["num_decoder_layers"],
            dropout=filtered_config["dropout"],
            batch_size=filtered_config["batch_size"],
            n_epochs=filtered_config["n_epochs"],
            random_state=filtered_config.get("random_state", 42)
        )

    def fit(self, series: TimeSeries):
        self.model.fit(series)

    def predict(self, n: int):
        return self.model.predict(n)