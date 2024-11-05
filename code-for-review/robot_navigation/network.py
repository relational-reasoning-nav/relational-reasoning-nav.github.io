import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
from robot_navigation.embedding import HumanHumanEdgeEmbed, HumanRobotEdgeEmbed, HumanNodeEmbed, EdgeAttention


@dataclass
class NetworkParams:
    human_node_embed_size: int = 64
    human_node_output_size: int = 64
    human_node_input_size: int = 32
    human_node_embedding_size: int = 64
    human_human_edge_embed_size: int = 64
    human_human_edge_embedding_size: int = 64
    human_human_edge_input_size: int = 32
    attention_size: int = 64
    seq_length: int = 20
    num_processes: int = 1
    num_mini_batch: int = 4
    no_cuda: bool = False


class SocialRobotNav(nn.Module):
    def __init__(
        self, 
        obs_space_dict: Dict[str, torch.Tensor], 
        params: NetworkParams,
        infer: bool = False
    ):
        super().__init__()
        self.infer = infer
        self.params = params
        self.human_num = obs_space_dict["spatial_edges"].shape[0]
        self.setup_networks(obs_space_dict)

    def setup_networks(self, obs_space_dict: Dict[str, torch.Tensor]):
        self.humanNodeEmbed = HumanNodeEmbed(
            self.params.human_node_embed_size,
            self.params.human_node_output_size,
            self.params.human_node_embedding_size,
            self.params.human_node_input_size,
            self.params.human_human_edge_embed_size,
        )

        spatial_input_size = obs_space_dict["spatial_edges"].shape[1]
        
        self.humanhumanEdgeEmbed_spatial = HumanHumanEdgeEmbed(
            self.params.human_human_edge_embed_size,
            self.params.human_human_edge_embedding_size,
            spatial_input_size,
        )
        
        self.humanhumanEdgeEmbed_temporal = HumanHumanEdgeEmbed(
            self.params.human_human_edge_embed_size,
            self.params.human_human_edge_embedding_size,
            self.params.human_human_edge_input_size,
        )

        self.humanrobotEdgeEmbed = HumanRobotEdgeEmbed(
            self.params.human_human_edge_embed_size,
            self.params.human_human_edge_embedding_size,
            self.params.human_robot_edge_input_size,
        )

        self.global_attn = EdgeAttention(
            self.params.human_human_edge_embed_size,
            self.params.human_node_embed_size,
            self.params.attention_size,
        )

        hidden_size = self.params.human_node_output_size
        self.feature_gru = nn.GRU(
            input_size=hidden_size * 3,  # Combined features from different embeddings
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        self.robot_linear = nn.Linear(7, self.params.human_node_input_size)
        self._init_weights()

    def _init_weights(self):
        for m in [self.actor, self.critic, self.robot_linear, self.feature_gru]:
            if isinstance(m, (nn.Linear, nn.GRU)):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0) if hasattr(m, 'bias') else None

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        rnn_hxs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        seq_length = 1 if self.infer else self.params.seq_length
        nenv = (
            self.params.num_processes
            if self.infer
            else self.params.num_processes // self.params.num_mini_batch
        )

        tensors = {
            "robot_node": self._reshape(inputs["robot_node"], seq_length, nenv),
            "temporal_edges": self._reshape(inputs["temporal_edges"], seq_length, nenv),
            "spatial_edges": self._reshape(inputs["spatial_edges"], seq_length, nenv),
            "robot_human_edges": self._reshape(inputs["robot_human_edges"], seq_length, nenv),
            "hidden_states_node": self._reshape(rnn_hxs["human_node_rnn"], 1, nenv),
            "hidden_states_edge": self._reshape(rnn_hxs["human_human_edge_rnn"], 1, nenv),
            "hidden_states_gru": self._reshape(rnn_hxs["feature_gru"], 1, nenv),
            "masks": self._reshape(masks, seq_length, nenv),
        }

        # Process human-human edges (temporal and spatial)
        output_temporal, _, _ = self.humanhumanEdgeEmbed_temporal(
            tensors["temporal_edges"],
            tensors["hidden_states_edge"],
            tensors["masks"],
        )

        output_spatial, _, _ = self.humanhumanEdgeEmbed_spatial(
            tensors["spatial_edges"],
            tensors["hidden_states_edge"],
            tensors["masks"],
        )

        # Process human-robot edges
        output_robot, _, _ = self.humanrobotEdgeEmbed(
            output_temporal,
            tensors["hidden_states_edge"],
            tensors["masks"],
        )

        hh_attention_output, _ = self.global_attn(output_temporal, output_spatial)
        hr_attention_output, _ = self.global_attn(output_robot, output_spatial)
        
        robot_features = self.robot_linear(tensors["robot_node"])
        
        combined_features = torch.cat([
            hh_attention_output,
            hr_attention_output,
            robot_features
        ], dim=-1)

        gru_output, gru_hidden = self.feature_gru(
            combined_features.view(-1, combined_features.size(-1)).unsqueeze(1),
            tensors["hidden_states_gru"]
        )

        # Update hidden states
        rnn_hxs.update({
            "feature_gru": gru_hidden,
        })

        features = gru_output.squeeze(1)
        critic_features = self.critic(features)
        actor_features = self.actor(features)

        if self.infer:
            return (
                critic_features.squeeze(),
                actor_features.squeeze(),
                rnn_hxs,
            )

        return (
            critic_features.view(-1, 1),
            actor_features.view(-1, self.params.human_node_output_size),
            rnn_hxs,
        )

    def _reshape(self, tensor: torch.Tensor, seq_length: int, nenv: int) -> torch.Tensor:
        return tensor.view(seq_length, nenv, *tensor.size()[1:])

    def _get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")