from typing import Dict, Any, List
from dataclasses import dataclass, field
import gym

from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import SACPolicy
from .backbones import BACKBONES
from .custome_logger import CustomeLogger
from .callbacks import BaseCallback, CheckpointCallback, RecordVideoCallback

from gym.spaces import Dict as DictSpace, Box


@dataclass
class SACConfig:
    #backbone: str = 'CNN'
    backbone_config: Dict[str, Any] = field(default_factory=dict)

    verbose: bool = True

    gamma: float = 0.95

    training_steps: int = 1000000

    log_interval: int = 4

    record_interval: int = 100
    save_freq: int = 50000

class SACAgent:
    def __init__(self, env: gym.Env, config: SACConfig) -> None:
        backbone = 'CNN' if isinstance(env.observation_space, Box) else 'PointNet'

        self.model = SAC(
            SACPolicy, 
            env,
            policy_kwargs=dict(
                features_extractor_class=BACKBONES[backbone],
                features_extractor_kwargs=config.backbone_config,
            ),
            verbose=config.verbose, 
            gamma = config.gamma
        )
        self.config = config


    def main(self, callbacks: List[BaseCallback]=[]):
        self.model.set_logger(CustomeLogger())
        callbacks += [
            CheckpointCallback(save_freq=self.config.save_freq),
            RecordVideoCallback(render_freq=self.config.record_interval),
        ]

        self.model.learn(
            self.config.training_steps, 
            callback=callbacks, 
            log_interval=self.config.log_interval,
            progress_bar=True,
        )