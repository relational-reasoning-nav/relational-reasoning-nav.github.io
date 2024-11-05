import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from typing import Dict, Tuple, List, Optional, Any


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        embed_size: int, 
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = q.size(0)
        
        Q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        out = self.out_linear(out)
        
        return out


class EdgeAttention(nn.Module):
    def __init__(self, edge_embed_size: int, node_embed_size: int, attention_size: int):
        super().__init__()
        self.attention_size = attention_size
        self.temporal_edge_layer = nn.ModuleList(
            [nn.Linear(edge_embed_size, attention_size)]
        )
        self.spatial_edge_layer = nn.ModuleList(
            [nn.Linear(edge_embed_size, attention_size)]
        )

    def forward(
        self, h_temporal: torch.Tensor, h_spatials: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_embed = self.temporal_edge_layer[0](h_temporal)
        spatial_embed = self.spatial_edge_layer[0](h_spatials)

        seq_len, nenv, _, h_size = h_spatials.size()
        attn = torch.sum(temporal_embed * spatial_embed, dim=3)
        temperature = h_spatials.size(2) / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)
        attn = torch.nn.functional.softmax(attn.view(seq_len, nenv, 1, -1), dim=-1)

        h_spatials_reshaped = (
            h_spatials.view(seq_len * nenv * 1, -1, h_size).permute(0, 2, 1)
        )
        attn_reshaped = attn.view(seq_len * nenv * 1, -1).unsqueeze(-1)
        weighted_value = (
            torch.bmm(h_spatials_reshaped, attn_reshaped)
            .squeeze(-1)
            .view(seq_len, nenv, 1, h_size)
        )

        return weighted_value, attn
