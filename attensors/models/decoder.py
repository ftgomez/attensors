import torch.nn as nn

from attensors.layers import Embeddings, MultiHeadAttention


class DecoderLayer(nn.Module):
    """
    Single layer of the Transformer Decoder

    Args:
        d_model (int): The dimension of the model
        num_heads (int): The number of attention heads
        ff_hidden_dim (int): The hidden dimension of the feedforward layer
        dropout (float): Dropout rate
    """

    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model),
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, enc_mask):
        """
        Forward pass of the DecoderLayer

        Args:
            x (torch.Tensor): Input tensor
            of shape (seq_len, batch_size, d_model)

            enc_output (torch.Tensor): Encoder output tensor
            of shape (seq_len, batch_size, d_model)

            self_mask (torch.Tensor): Mask tensor for self-attention
            of shape (batch_size, num_heads, seq_len, seq_len)

            enc_mask (torch.Tensor): Mask tensor for encoder-decoder attention
            of shape (batch_size, num_heads, seq_len_enc, seq_len_dec)

        Returns:
            torch.Tensor: Output tensor
            of shape (seq_len, batch_size, d_model)
        """
        self_attention_output, _ = self.self_attention(x, x, x, self_mask)
        self_attention_output = self.dropout(self_attention_output)
        x = self.layer_norm1(x + self_attention_output)
        enc_dec_attention_output, _ = self.encoder_attention(
            x, enc_output, enc_output, enc_mask
        )
        enc_dec_attention_output = self.dropout(enc_dec_attention_output)
        x = self.layer_norm2(x + enc_dec_attention_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.layer_norm3(x + ffn_output)

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder module with embeddings and positional encoding

    Args:
        num_layers (int): Number of decoder layers
        d_model (int): The dimension of the model
        num_heads (int): The number of attention heads
        ff_hidden_dim (int): The hidden dimension of the feedforward layer
        dropout (float): Dropout rate
        vocab_size (int): Size of the vocabulary
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        ff_hidden_dim,
        dropout=0.1,
        vocab_size=None,
    ):
        super(Decoder, self).__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, enc_output, self_mask, enc_mask):
        """
        Forward pass of the Decoder module

        Args:
            x (torch.Tensor): Input tensor
            of shape (seq_len, batch_size)

            enc_output (torch.Tensor): Encoder output tensor
            of shape (seq_len, batch_size, d_model)

            self_mask (torch.Tensor): Mask tensor for self-attention
            of shape (batch_size, num_heads, seq_len, seq_len)

            enc_mask (torch.Tensor): Mask tensor for encoder-decoder attention
            of shape (batch_size, num_heads, seq_len_enc, seq_len_dec)

        Returns:
            torch.Tensor: Output tensor
            of shape (seq_len, batch_size, d_model)
        """
        embedded = self.embedding(x)
        x = embedded
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, self_mask, enc_mask)
        return x
