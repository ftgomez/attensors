import torch.nn as nn

from attensors.layers import Embeddings, MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    Single layer of the Transformer Encoder

    Args:
        d_model (int): The dimension of the model
        num_heads (int): The number of attention heads
        ff_hidden_dim (int): The hidden dimension of the feedforward layer
        dropout (float): Dropout rate
    """

    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model),
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass of the EncoderLayer

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)
            mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model)
        """
        attention_output, _ = self.self_attention(x, x, x, mask)
        attention_output = self.dropout(attention_output)
        x = self.layer_norm1(x + attention_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.layer_norm2(x + ffn_output)

        return x


class Encoder(nn.Module):
    """
    Transformer Encoder module with embeddings and positional encoding

    Args:
        num_layers (int): Number of encoder layers
        d_model (int): The dimension of the model
        num_heads (int): The number of attention heads
        ff_hidden_dim (int): The hidden dimension of the feedforward layer
        dropout (float): Dropout rate
        max_len (int): Maximum sequence length for positional encoding
        vocab_size (int): Size of the vocabulary
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        ff_hidden_dim,
        dropout=0.1,
        max_len=5000,
        vocab_size=None,
    ):
        super(Encoder, self).__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        """
        Forward pass of the Encoder module

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size)
            mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model)
        """
        embedded = self.embedding(x)
        x = embedded
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x
