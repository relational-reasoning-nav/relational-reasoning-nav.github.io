import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
from robot_navigation.attn import MultiHeadAttention


class HumanHumanEdgeEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding_size: int,
        input_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        
        self.encoder_linear = nn.Linear(input_size, embedding_size)
        self.attention = MultiHeadAttention(embedding_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_size),
            nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()

    def forward(
        self, 
        inp: torch.Tensor, 
        h: torch.Tensor, 
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len, nenv, agent_num, _ = inp.size()
        encoded_input = self.relu(self.encoder_linear(inp))
        
        reshaped_input = encoded_input.view(seq_len * nenv, agent_num, -1)
        reshaped_masks = masks.view(seq_len * nenv, 1, 1)
        
        attn_out = self.attention(
            reshaped_input, reshaped_input, reshaped_input, reshaped_masks
        )
        out1 = self.norm1(reshaped_input + attn_out)
        
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_out)
        
        outputs = out2.view(seq_len, nenv, agent_num, -1)
        return outputs, encoded_input, outputs


class HumanRobotEdgeEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding_size: int,
        input_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        
        self.encoder_linear = nn.Linear(input_size, embedding_size)
        self.attention = MultiHeadAttention(embedding_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_size),
            nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()

    def forward(
        self, 
        inputs: torch.Tensor, 
        hidden: torch.Tensor, 
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len, nenv, agent_num, _ = inputs.size()
        encoded = self.relu(self.encoder_linear(inputs))
        
        reshaped_input = encoded.view(seq_len * nenv, agent_num, -1)
        reshaped_masks = masks.view(seq_len * nenv, 1, 1)
        
        attn_out = self.attention(
            reshaped_input, reshaped_input, reshaped_input, reshaped_masks
        )
        out1 = self.norm1(reshaped_input + attn_out)
        
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_out)
        
        outputs = out2.view(seq_len, nenv, agent_num, -1)
        return outputs, encoded, outputs


class HumanNodeEmbed(nn.Module):
    def __init__(
        self,
        rnn_size: int,
        output_size: int,
        embedding_size: int,
        input_size: int,
        edge_rnn_size: int,
    ):
        super().__init__()
        self.rnn_size = rnn_size
        self.gru = nn.GRU(embedding_size * 2, rnn_size)
        self.encoder_linear = nn.Linear(input_size, embedding_size)
        self.edge_attention_embed = nn.Linear(edge_rnn_size * 2, embedding_size)
        self.output_linear = nn.Linear(rnn_size, output_size)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def forward(
        self,
        pos: torch.Tensor,
        h_temporal: torch.Tensor,
        h_spatial_other: torch.Tensor,
        h: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_input = self.relu(self.encoder_linear(pos))
        h_edges = torch.cat((h_temporal, h_spatial_other), dim=-1)
        h_edges_embedded = self.relu(self.edge_attention_embed(h_edges))
        concat_encoded = torch.cat((encoded_input, h_edges_embedded), dim=-1)

        x, h_new = self._forward_sequence(concat_encoded, h, masks)
        outputs = self.output_linear(x)
        return outputs, h_new

    def _forward_sequence(
        self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len, nenv, agent_num, _ = x.size()
        x_flat = x.view(seq_len, nenv * agent_num, -1)

        if x.size(0) == hxs.size(0):
            masks = masks.view(x.size(0), -1, 1)
            hxs = (hxs * masks).view(x.size(0), -1, hxs.size(-1))
            outputs, hxs_new = self.gru(x_flat, hxs)
            return outputs.view_as(x), hxs_new.view_as(hxs)

        T, N = x.size(0), x.size(1)
        masks = masks.view(T, N)
        has_zeros = (masks[1:] == 0).any(dim=-1).nonzero().squeeze().cpu()
        has_zeros = [0] + (has_zeros + 1).tolist() + [T]
        outputs = []

        for i in range(len(has_zeros) - 1):
            start, end = has_zeros[i], has_zeros[i + 1]
            cur_x = x_flat[start:end]
            mask = masks[start]
            
            mask = mask.view(1, -1, 1)
            cur_hxs = (hxs * mask).view(hxs.size(0), N * agent_num, -1)
            rnn_scores, hxs = self.gru(cur_x, cur_hxs)
            outputs.append(rnn_scores)

        outputs = torch.cat(outputs, dim=0)
        return outputs.view(T, N, agent_num, -1), hxs.view(1, N, agent_num, -1)