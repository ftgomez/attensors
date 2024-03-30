import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer

    Args:
        d_model (int): The dimension of the model
        max_len (int): The maximum sequence length
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            torch.Tensor: Output tensor with positional encodings added,
            shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Embeddings(nn.Module):
    """
    Embedding Layer for Transformer

    Args:
        vocab_size (int): The size of the vocabulary
        d_model (int): The dimension of the model
        max_len (int): The maximum sequence length
    """

    def __init__(self, vocab_size, d_model, max_len=5000):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor representing token indices
            of shape (seq_len, batch_size)

        Returns:
            torch.Tensor: Embedded tensor with positional encodings added,
            shape (seq_len, batch_size, d_model)
        """
        embedded = self.embedding(x)
        embedded_with_position = self.positional_encoding(embedded)
        return embedded_with_position
