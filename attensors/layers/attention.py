import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for Transformer

    Args:
        d_model (int): The dimension of the model
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, embedding, mask=None):
        """
        Args:
            embedding (torch.Tensor): Embedding tensor
            of shape (seq_len, batch_size, d_model)

            mask (torch.Tensor): Mask tensor indicating
            which elements to mask out (optional)

        Returns:
            torch.Tensor: Output tensor
            of shape (seq_len, batch_size, d_model)

            and attention weights tensor
            of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = embedding.shape[1]

        Q = self.query(embedding)
        K = self.key(embedding)
        V = self.value(embedding)

        Q = Q.view(-1, batch_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(-1, batch_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(-1, batch_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention_weights = F.softmax(energy, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        output = (
            output.permute(0, 2, 1, 3).contiguous().view(-1, batch_size, self.d_model)
        )
        output = self.fc_out(output)

        return output, attention_weights
