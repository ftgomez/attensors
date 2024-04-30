import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for Transformer

    Args:
        d_model (int): The dimension of the model
        num_heads (int): The number of attention heads
        dropout (float): Dropout rate
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim)

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)
            batch_size (int): Batch size

        Returns:
            torch.Tensor: Split tensor
            of shape (batch_size, num_heads, seq_len, head_dim)
        """
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the Multi-Head Attention module

        Args:
            query (torch.Tensor): Query tensor of shape (seq_len_q, batch_size, d_model)
            key (torch.Tensor): Key tensor of shape (seq_len_k, batch_size, d_model)
            value (torch.Tensor): Value tensor of shape (seq_len_v, batch_size, d_model)
            mask (torch.Tensor, optional): Mask tensor
            of shape (batch_size, seq_len_q, seq_len_k). Defaults to None.

        Returns:
            torch.Tensor: Output tensor
            of shape (seq_len_q, batch_size, d_model)

            torch.Tensor: Attention weights
            of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(1)

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = nn.functional.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)

        output = output.transpose(1, 2).contiguous().view(-1, batch_size, self.d_model)

        output = self.output_linear(output)

        return output, attention_weights
