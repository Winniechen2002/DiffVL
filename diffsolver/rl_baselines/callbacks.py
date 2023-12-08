import time
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.sac.policies import SACPolicy
from tools.utils import logger
from .custome_logger import CustomeLogger
import numpy as np
import wandb


class CollectTrainingInfo(BaseCallback):
    def _on_step(self) -> bool:
        if any(self.locals['dones']):
            logger: CustomeLogger = self.locals['self'].logger
            for i in self.locals['infos']:
                if 'evaluation' in i:
                    for k, v in i['evaluation'].items():
                        for name, val in v.items():
                            if name == '_images':
                                for name, img in val.items():
                                    logger.savefig(f'{k}_{name}.png', img)
                            else:
                                logger.record_mean(f'eval/{k}_{name}', val)
        return True


class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        name_prefix: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.name_prefix = name_prefix

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        return f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}"

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            env = self.model.env
            self.model.env = None
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(os.path.join(logger.get_dir(), model_path))

            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")
            self.model.env = env
        return True
        
class RecordVideoCallback(BaseCallback):
    def __init__(self, render_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.epoch_count = 0
        self.images = []

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.epoch_count += 1
            if len(self.images) > 0:
                logger.animate(self.images, 'video.mp4', fps=20)
                self.images = []

        if self.epoch_count % self.render_freq == 0:
            image = self.locals['env'].render(mode='rgb_array')
            self.images.append(image)
        return True