import torch
from envs import MultiToolEnv
from ..program.types.constraints import Constraint
from torch import nn
from torch.optim import Adam
from diffsolver.config import OptimConfig


class Optimizer:
    def __init__(
        self, 
        env: MultiToolEnv, 
        init_action: torch.Tensor, 
        config: OptimConfig
    ) -> None:
        self.env = env
        self.parameters =  nn.Parameter(init_action, requires_grad=True)
        self.optim = Adam([self.parameters], lr=config.lr)

    def step(self, constraint: Constraint):
        constraint.loss.backward()
        self.optim.step()
        self.parameters.data.clamp_(-1, 1)

    def __enter__(self):
        self.optim.zero_grad()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass