import torch
from torch import nn
import torch.nn.functional as F
import math

class SegFormerEncoder(nn.Module):
    #chatGPT...test it
    def __init__(self, in_channels, hidden_size, num_layers, num_heads, feedforward_size, dropout):
        super().__init__()

        # Input Embedding
        self.conv = nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_size * 64 * 64, hidden_size)

        # Positional Encoding
        self.pos_enc = PositionalEncoding(hidden_size, dropout)

        # Transformer Encoder Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerEncoderLayer(hidden_size, num_heads, feedforward_size, dropout))

        # Skip Connections
        self.skip_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.skip_convs.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=1))

    def forward(self, x):
        # Input Embedding
        x = self.conv(x)
        x = x.flatten(start_dim=2)
        x = self.fc(x)

        # Positional Encoding
        x = self.pos_enc(x)

        # Transformer Encoder Layers
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)

        # Skip Connections
        for i, skip_conv in enumerate(self.skip_convs):
            x = x + skip_conv(skip_connections[i])

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, max_length=4096):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.GELU(),
            nn.Linear(feedforward_size, hidden_size),
            nn.Dropout(p=dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = x.permute(1, 0, 2)
        x, attn_weights = self.self_attn(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = residual + x
        return x, attn_weights
