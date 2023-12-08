from diffsolver import set_render_devices #  please keep this line so that it can work on different machines
import os
import gym
import datetime
import random
import numpy as np

import torch
from typing import Dict, Any, cast, List

from omegaconf import OmegaConf, DictConfig
import argparse
from dataclasses import dataclass, field
from tools.utils import logger

from diffsolver.rl_baselines.sac import SACAgent, SACConfig
from diffsolver.rl_baselines.ppo import PPOAgent, PPOConfig
from diffsolver.rl_baselines.callbacks import CollectTrainingInfo
from diffsolver.utils import get_path


@dataclass
class EnvCfg:
    env_name: str = 'Lift-v0'
    config: Dict[str, Any] = field(default_factory=dict) 

@dataclass
class MainConfig:
    path: str
    seed: int
    config_id: str

    group_name: str|None = None 

    logger_format: List[str] = field(default_factory=lambda: ['stdout', 'csv'])
    use_wandb: bool = False

    env_cfg: EnvCfg = field(default_factory=EnvCfg)

    rl: str = 'sac'
    sac: SACConfig = field(default_factory=SACConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    # sac configs..


def make(env_cfg: EnvCfg):
    from diffsolver.rl_baselines.rl_envs import make # register env
    return gym.make(env_cfg.env_name, **env_cfg.config)

def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available

def get_group_name(env, agent, cfg: MainConfig):
    env_name = cfg.env_cfg.env_name.split('-')[0]
    if hasattr(env, 'get_suffix'):
        env_name += env.get_suffix()
    algo_name = ''
    if hasattr(agent, 'get_suffix'):
        algo_name = '_' + agent.get_suffix()

    now = datetime.datetime.now()
    group_name = env_name + algo_name + '_' + now.strftime("%m%d")
    group_name += '_' + cfg.config_id
    group_name = group_name 
    return group_name


def main(env: gym.Env|None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args, unknown = parser.parse_known_args()

    inp_cfg: DictConfig = OmegaConf.structured(MainConfig)
    if args.config is not None:
        inp_cfg.merge_with(OmegaConf.load(args.config))

    input_cfg = OmegaConf.from_dotlist(unknown)
    inp_cfg.merge_with(input_cfg)
    print(OmegaConf.to_yaml(inp_cfg))

    cfg = cast(MainConfig, inp_cfg)


    # seeding
    if not OmegaConf.is_missing(cfg, 'seed'):
        set_seed(cfg.seed)
        runid = f'seed{cfg.seed}'
    else:
        now = datetime.datetime.now()
        runid = now.strftime("%Y%m%d%H%M%S")
    
    env = env or make(cfg.env_cfg)

    if cfg.rl == 'ppo':
        agent = PPOAgent(env, cfg.ppo)
    elif cfg.rl == 'sac':
        agent = SACAgent(env, cfg.sac)
    else:
        raise NotImplementedError


    # naming
    if OmegaConf.is_missing(cfg, 'path'):
        if OmegaConf.is_missing(cfg, 'config_id'):
            assert args.config is not None, "Plase specify a config file or providing path"
            cast(DictConfig, cfg).merge_with(dict(config_id=args.config.split('/')[-1].split('.')[0]))
        group = get_group_name(env, SACAgent(env, cfg.sac), cfg)
        cast(DictConfig, cfg).merge_with(dict(path=get_path('MODEL_DIR', group, runid)))
    else:
        group = cfg.path.split('/')[-1]

    # logging
    kwargs = {}
    if cfg.use_wandb:
        kwargs = {'project': os.environ.get("WANDB_PROJECT_NAME", 'diffsolver'), 'group': cfg.group_name or group, 'name': runid, 'config': cfg}
        cfg.logger_format += ['wandb']
    logger.configure(cfg.path, cfg.logger_format, **kwargs)

    # main
    agent.main(callbacks=[CollectTrainingInfo()])



if __name__ == '__main__':
    with torch.device('cuda:0'):
        main()