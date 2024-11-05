import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
from robot_navigation.network import NetworkParams, SocialRobotNav


@dataclass
class PolicyParams:
    action_space: Any
    obs_space_dict: Dict[str, torch.Tensor]
    network_params: NetworkParams
    deterministic: bool = False


class SocialNavigationPolicy(nn.Module):
    def __init__(self, params: PolicyParams):
        super().__init__()
        self.params = params
        self.setup_policy()
        
    def setup_policy(self):
        self.network = SocialRobotNav(
            self.params.obs_space_dict,
            self.params.network_params,
        )

        action_space = self.params.action_space
        output_size = self.params.network_params.human_node_output_size

        if action_space.__class__.__name__ == "Discrete":
            self.dist = Categorical(output_size, action_space.n)
        elif action_space.__class__.__name__ == "Box":
            self.dist = DiagGaussian(output_size, action_space.shape[0])
        elif action_space.__class__.__name__ == "MultiBinary":
            self.dist = Bernoulli(output_size, action_space.shape[0])
        else:
            raise NotImplementedError(f"Unsupported action space: {action_space.__class__.__name__}")

    @property
    def is_recurrent(self) -> bool:
        return True

    @property
    def recurrent_hidden_state_size(self) -> int:
        return self.params.network_params.human_node_embedding_size

    def act(
        self,
        inputs: Dict[str, torch.Tensor],
        rnn_hxs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        value, actor_features, rnn_hxs = self.network(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(
        self,
        inputs: Dict[str, torch.Tensor],
        rnn_hxs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        value, _, _ = self.network(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(
        self,
        inputs: Dict[str, torch.Tensor],
        rnn_hxs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        value, actor_features, rnn_hxs = self.network(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        return value, action_log_probs, dist_entropy, rnn_hxs

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        rnn_hxs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return self.network(inputs, rnn_hxs, masks)


class Categorical(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, 0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return torch.distributions.Categorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = nn.Parameter(torch.zeros(num_outputs))
        
        nn.init.orthogonal_(self.mean.weight, 1)
        nn.init.constant_(self.mean.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean(x)
        std = self.logstd.exp()
        return torch.distributions.Normal(mean, std)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, 1)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return torch.distributions.Bernoulli(logits=x)
