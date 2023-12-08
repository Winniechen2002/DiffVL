# https://github.com/hzaskywalker/RPGv2/blob/rpg/envs/softbody/plb_envs.p
import cv2
import gym
import torch
import sapien.core as sapien
import numpy as np

#from .camera import CameraGUI
from diffsolver.utils.renderer import CameraGUI
import numpy as np
from tools.config import CN
from gym.spaces import Box, Dict

from typing import cast, Any
from diffsolver.paths import get_path, touch, DICTS
from diffsolver.config import SceneConfig, ToolSamplerConfig
from diffsolver.program.scenes import load_scene_with_envs
from diffsolver.toolguide import ToolSampler
from diffsolver.program.types import SceneSpec
from diffsolver.program.prog_registry import _PROG_LIBS
from diffsolver.program import progs # noqa: F401
from omegaconf import OmegaConf, DictConfig
from envs import MultiToolEnv
from dataclasses import dataclass, field
from typing import List

from gym.wrappers.time_limit import TimeLimit
from diffsolver.config import EvalConfig
from diffsolver.evaluator.evaluator import Evaluator


DICTS['RLENVS'] = touch(get_path('ASSET_DIR', 'rl_envs'))

def subsample_pcd(n: int, *args: torch.Tensor) -> List[torch.Tensor]:
    label = torch.randint(len(args[0]), (n,), device=args[0].device)
    out = [i[label] for i in args]
    return out

@dataclass
class LookatConfig:
    center: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.2])
    theta: float = 0.
    phi: float = np.pi/4
    radius: float = 2.5
    zeta: float = 0


@dataclass
class RLConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    lookat: LookatConfig = field(default_factory=LookatConfig)

    evaluator: EvalConfig = field(default_factory=EvalConfig)

    prog: str = 'lift()'
    max_eipisode_steps: int = 100

    obs_mode: str = 'rgb'
    n_pcd: int = 256

        
class GymEnv(gym.Env):
    def __init__(
        self, 
        config: RLConfig
    ) -> None:
        super().__init__()


        # import os
        #RENDER_DEVICES = os.environ.get('RENDER_DEVICES', '0')
        # assert "CUDA_VISIBLE_DEVICES" in os.environ and RENDER_DEVICES == os.environ['CUDA_VISIBLE_DEVICES'], "CUDA_VISIBLE_DEVICES must be the same as RENDER_DEVICES. Please set them by importing set_render_devices in the beginning."

        self.gui = CameraGUI(offscreen=True)
        self.gui.lookat(config.lookat)
        self.env = MultiToolEnv(sim_cfg=CN(dict(max_steps=100)))

        self.scene_tuple = load_scene_with_envs(self.env, config.scene)
        self.tool_sampler = ToolSampler(self.env, ToolSamplerConfig(n_samples=0))
        self.scene = SceneSpec.from_scene_tuple(
            self.env, self.scene_tuple, self.tool_sampler.select_tool(self.scene_tuple)
        )

        color = self.scene.state.color
        assert color is not None
        b = color % 0xFF
        g = (color // 256) % 0xFF
        r = (color // 65536) % 0xFF
        color_rgb = np.stack([r, g, b], axis=-1)

        self.color_tensor = torch.tensor(color_rgb, dtype=torch.float32, device=self.env.device) / 255


        self.rewards = _PROG_LIBS.parse(config.prog)
        self.evaluator = Evaluator(self.scene, config.evaluator)
        self.obs_mode = config.obs_mode
        assert self.obs_mode in ['rgb', 'pcd']
        self.config = config

        obs = self.reset()
        if self.obs_mode == 'rgb':
            assert isinstance(obs, np.ndarray)
            self.observation_space = Box(low=0, high=255, shape=obs.shape, dtype=np.uint8)
        else:
            assert isinstance(obs, dict)
            self.observation_space = Dict(spaces={
                k: Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32) for k, v in obs.items()
            })
        self.action_space = self.env.action_space

    def get_obs(self):
        #TODO: add tool pcd if necessary ..
        if self.obs_mode == 'rgb':
            return cv2.resize(self.render(), (84, 84))
        else:
            obs = self.env.get_obs()
            points = torch.concat([obs['pos'], obs['vel'], self.color_tensor/5.], dim=0)
            return{
                'pcd': subsample_pcd(self.config.n_pcd, points)[0].detach().cpu().numpy(),
                'agent': obs['qpos'].detach().cpu().numpy()
            }

    def reset(self):
        self.env.set_state(self.scene_tuple.state)
        self.gui.reset(self.env)
        obs = self.get_obs()
        self.evaluator.reset(self.env.get_obs())
        return obs


    def step(self,action):
        obs = self.env.step(action)[0]

        self.scene.obs = [obs]
        constraints = self.rewards(self.scene)
        reward = - constraints.loss.item()
        self.evaluator.onstep(obs, torch.tensor(action, dtype=torch.float32), locals())
        return self.get_obs(), reward, False, {}

    def render(self, mode='rgb_array'):
        return self.gui.capture()

    def get_suffix(self):
        return '_' + self.obs_mode

class TimeLimit2(TimeLimit):
    # modify it so that we can call the evaluator in the ending state
    def step(self, action):
        obs, r, done, info = super().step(action)
        if done: # done
            assert isinstance(self.env, GymEnv)
            info['evaluation'] = self.env.evaluator.onfinish()
        return obs, r, done, info
        

def make(config_path=None, **kwargs):
    config: RLConfig = OmegaConf.structured(RLConfig)

    if config_path is not None:
        cast(DictConfig, config).merge_with(OmegaConf.load(get_path('RLENVS', config_path)))

    if len(kwargs) > 0:
        cast(DictConfig, config).merge_with(OmegaConf.create(kwargs))
    
    return TimeLimit2(GymEnv(config), max_episode_steps=config.max_eipisode_steps)