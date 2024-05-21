import torch.nn as nn

from attensors.layers import Embeddings, MultiHeadAttention


class GPTBlock(nn.Module):
    """
    Single layer of the GPT architecture

    Args:
        d_model (int): The dimension of the model
        num_heads (int): The number of attention heads
        ff_hidden_dim (int): The hidden dimension of the feedforward layer
        dropout (float): Dropout rate
    """

    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(GPTBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
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
        Forward pass of the GPTBlock

        Args:
            x (torch.Tensor): Input tensor
            of shape (seq_len, batch_size, d_model)

            mask (torch.Tensor): Mask tensor for decoder attention
            of shape (batch_size, num_heads, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor
            of shape (seq_len, batch_size, d_model)
        """
        attention_output, _ = self.attention(x, x, x, mask)
        self_attention_output = self.dropout(attention_output)
        x = self.layer_norm1(x + self_attention_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.layer_norm2(x + ffn_output)

        return x


class GPT(nn.Module):
    """
    GPT model

    Args:
        num_layers (int): Number of decoder layers
        d_model (int): The dimension of the model
        num_heads (int): The number of attention heads
        ff_hidden_dim (int): The hidden dimension of the feedforward layer
        dropout (float): Dropout rate
        vocab_size (int): Size of the vocabulary
        max_len (int): Max sequence length
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        ff_hidden_dim,
        dropout=0.1,
        vocab_size=None,
        max_len=5000,
    ):
        super(GPT, self).__init__()
        self.embedding = Embeddings(vocab_size, d_model, max_len)
        self.gpt_blocks = nn.ModuleList(
            [
                GPTBlock(d_model, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.max_len = max_len

    def forward(self, trg, trg_mask):
        """
        Forward pass of the GPT model.

        Args:
            trg (torch.Tensor): Input tensor to the decoder representing token indices.
            trg_mask (torch.Tensor): Mask tensor for target sequence.

        Returns:
            torch.Tensor: Output tensor from the decoder.
            Shape (seq_len, batch_size, vocab_size)
        """

        assert (
            trg.size(0) <= self.max_len
        ), "The input sequence length exceeds the maximum length allowed"

        x = self.embedding(trg)

        for gpt_block in self.gpt_blocks:
            x = gpt_block(x, trg_mask)

        x = self.output_linear(x)
        x = self.softmax(x)

        return x
