from typing import List
from dataclasses import dataclass, field
from omegaconf import OmegaConf

@dataclass
class GymEnvConfig:
    alpha: float = 0.1
    beta: float = 10.0
    gamma: float = 0.2
    kappa: float = 1.0
    zeta: float = 0.1
    epsilon: float = 0.02
    task: int = 1
    scene_id: int = 1
    target_scene_id: int = 1
    sim_max_step: int = 100
    samples: int = 200
    voxel_size: float = 0.01

@dataclass
class PointNetConfig:
    features_dim: int = 128
    hidden_size: int = 1024