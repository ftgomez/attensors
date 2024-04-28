import torch.nn as nn

from attensors.layers import Embeddings, MultiHeadAttention, PositionalEncoding


class EncoderLayer(nn.Module):
    """
    Encoder layer of the Transformer model.

    Args:
        d_model (int): The dimension of the model.
        num_heads (int): Number of attention heads.
        ff_hidden_dim (int): Hidden dimension of the feed-forward layer.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
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
        Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            mask (torch.Tensor): Mask tensor indicating which elements to mask out.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        attention_output, _ = self.self_attention(x, mask=mask)
        x = self.layer_norm1(x + self.dropout(attention_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x


class Encoder(nn.Module):
    """
    Transformer Encoder module.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the model.
        num_layers (int): Number of encoder layers.
        num_heads (int): Number of attention heads.
        ff_hidden_dim (int): Hidden dimension of the feed-forward layer.
        max_len (int): The maximum sequence length.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        ff_hidden_dim,
        max_len=5000,
        dropout=0.1,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = Embeddings(vocab_size, d_model, max_len)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, mask=None):
        """
        Forward pass of the encoder.

        Args:
            src (torch.Tensor): Input tensor representing
            token indices of shape (seq_len, batch_size).

            mask (torch.Tensor): Mask tensor indicating
            which elements to mask out (optional).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        x = self.embeddings(src)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
