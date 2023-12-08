# a fake 
from tabulate import tabulate
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import cast, TypedDict, MutableMapping, Tuple
from .optimizer import Optimizer, OptimConfig, MultiToolEnv, Constraint


@dataclass
class FrankWolfeConfig:
    lmbda_lr: float = 1.
    reg_prog: float = 0.01
    weight_penalty: float = 0.001
    constraint_threshold: float = 0.
    clip_lambda: float|None = None

    mu: float = 1.

class Summary(TypedDict):
    loss: float
    constraint: float
    penalty: float
    unscaled_constraint: float


class OutputInfo(TypedDict):
    summary: Summary
    constraints: MutableMapping[str, Tuple[float, float]]
    losses: MutableMapping[str, float]


def to_table(output_info: OutputInfo) -> str:
    # Extract data and create the table
    table_data = [
        ["Summary", "Loss", output_info["summary"]["loss"]],
        ["", "Constraint", output_info["summary"]["constraint"]],
        ["", "Penalty", output_info["summary"]["penalty"]],
        ["", "Unscaled Constraint", output_info["summary"]["unscaled_constraint"]],
    ]

    for idx, (name, constraint) in enumerate(output_info["constraints"].items()):
        table_data.append(["Constraints" if idx == 0 else "", name, f"{constraint[0]}, {constraint[1]}"])

    for idx, (name, loss) in enumerate(output_info["losses"].items()):
        table_data.append(["Losses" if idx == 0 else "", name, loss])

    # Print the table
    return tabulate(table_data, headers=["Category", "Name", "Value"])

class FrankWolfe(Optimizer):
    # this is not a frankwolfe algorithm ..
    # do project: https://vene.ro/blog/mirror-descent.html
    def __init__(
        self, 
        env: MultiToolEnv, 
        init_action: torch.Tensor, 
        config: OptimConfig
    ):
        self.env = env

        self.lr = config.lr
        self.frank_config = cast(FrankWolfeConfig, OmegaConf.merge(OmegaConf.structured(FrankWolfeConfig), config.frank_wolfe))


        self.parameters =  nn.Parameter(init_action, requires_grad=True)
        self.reward_optim = torch.optim.Adam([self.parameters], lr=self.lr)
        self.constraint_optim = torch.optim.Adam([self.parameters], lr=self.lr)

        self.lmbda = {}
        self.last_good = None

    def __enter__(self):
        return self

    def step(self, outputs: Constraint):
        """
        Loss: the criterion to optimize
        Constraint: unless the constraint is satisfied, the loss is not optimized
        Penalty: the penalty for the constraint
        unscaled_constraint: the constraint value before scaling by lambda

        self.lambda: the lagrangian multiplier for the constraint
        """

        loss, constraint, penalty, unscaled_constraint = 0., 0., 0., 0.


        losses = {}
        constraints = {}

        for cons in outputs.all():
            value = cons.loss
            key = cons.code
            if cons.is_constraint:
                # strict constraint

                if key not in self.lmbda:
                    self.lmbda[key] = 1.
                
                if value > 0.:
                    constraint = constraint + value * self.lmbda[key] + value * value * self.frank_config.mu/2
                    unscaled_constraint += value
                elif value < 0.:
                    penalty = penalty - torch.log(-value) * self.frank_config.weight_penalty

                self.lmbda[key] = max(self.lmbda[key] + self.frank_config.lmbda_lr * float(value), 0.5)
                if self.frank_config.clip_lambda is not None:
                    self.lmbda[key] = min(self.lmbda[key], self.frank_config.clip_lambda)

                constraints[key] = (float(cons.loss), float(self.lmbda[key]))

            else:
                loss += torch.relu(value)
                losses[key] = (float(value))


        summary = Summary(constraint=float(constraint), loss=float(loss), penalty=float(penalty), unscaled_constraint=float(unscaled_constraint))
        tables:  OutputInfo = {
            'summary': summary,
            'losses': losses,
            'constraints': constraints,
        }
        print(to_table(tables))


        action_regularizer = 0.
        if constraint <= self.frank_config.constraint_threshold:
            with torch.no_grad():
                self.last_good = self.parameters.clone()

            if isinstance(loss, torch.Tensor) or isinstance(penalty, torch.Tensor):
                self.reward_optim.zero_grad()
                val = loss + penalty
                assert isinstance(val, torch.Tensor)
                val.backward()
                self.reward_optim.step()

            action_regularizer = 0.
            
        else:
            self.constraint_optim.zero_grad()
            if self.frank_config.reg_prog > 0. and self.last_good is not None:
                action_regularizer = torch.linalg.norm(self.parameters - self.last_good)**2 * self.frank_config.reg_prog
            constraint += action_regularizer
            assert isinstance(constraint, torch.Tensor)
            constraint.backward()
            self.constraint_optim.step()

        with torch.no_grad():
            self.parameters.data.clamp_(-1, 1)


        return {
            'loss': float(loss),
            'creteria': float(unscaled_constraint) * 1000 + float(loss)
        }
            