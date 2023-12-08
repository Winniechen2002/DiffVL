import torch
from torch import nn
from gym.spaces import Dict, Box
from typing import Tuple
from dataclasses import dataclass, field
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict


@dataclass
class PointNetConfig:
    pn_dims: Tuple[int, ...] = field(default_factory=lambda : (32, 64))
    agent_dims: Tuple[int, ...] = field(default_factory=lambda : ())
    mlp: Tuple[int, ...] = field(default_factory=lambda : (256, 256))
    feature_dim: int = 256
    pooling_method: str = 'max'


def mlp(inp_dim: int, dims: Tuple[int, ...], out_dim: int):
    layers = []
    for dim in dims:
        layers.append(nn.Linear(inp_dim, dim))
        layers.append(nn.ReLU())
        inp_dim = dim
    layers.append(nn.Linear(inp_dim, out_dim))
    return nn.Sequential(*layers)

class PointNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict, feature_dims: int=512, **kwargs) -> None:
        super().__init__(observation_space, feature_dims)
        config = PointNetConfig(**kwargs)

        pcd_space = observation_space.spaces['pcd']
        agent_space = observation_space.spaces['agent']
        assert isinstance(pcd_space, Box)
        assert isinstance(agent_space, Box)
        assert config.pooling_method in ['max', 'mean']

        self.pcd_N = pcd_space.shape[0]
        self.pcd_dim = pcd_space.shape[1]
        self.agent_dim = agent_space.shape[0]

        self.pcd_mlp = mlp(self.pcd_dim, config.pn_dims, config.feature_dim)
        self.agent_mlp = mlp(self.agent_dim, config.agent_dims, config.feature_dim)
        self.combiner = mlp(config.feature_dim, config.mlp, feature_dims)
        self.config = config

    def forward(self, inp: TensorDict):
        pcd: torch.Tensor = self.pcd_mlp(inp['pcd'])
        assert pcd.shape[-2] == self.pcd_N
        pcd = pcd.max(dim=-2).values if self.config.pooling_method == 'max' else pcd.mean(dim=-2)
        agent = self.agent_mlp(inp['agent'])
        return self.combiner(pcd + agent)