# python3 run_single_stage.py prog --config examples/single_stage/task36_rope.yaml
# saving to MODEL_DIR/task_cfg.path
import sys
import numpy as np
import os
import json
import datetime
from typing import cast
import pandas as pd
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
from enum import Enum
import tempfile
import argparse
from diffsolver.config import DefaultConfig

FILEPATH = os.path.dirname(os.path.abspath(__file__))


class Task(Enum):
    prog = 'prog' # use huamn-written program
    lang = 'lang' # use language for both diffphys and tool sampling
    solve_phys = 'solve_phys' # use GPT generated program to solve physics
    solve_tool = 'solve_tool' # use GPT generated program to sample tools
    solve_phys_dev = 'solve_phys_dev'
    solve_lang_dev = 'solve_lang_dev' # use GPT generated program to solve physics using v2

    sac = 'sac' # run RL
    ppo = 'ppo' # run ppo
    oracle = 'oracle' # run oracle
    emdonly = 'emdonly' # run emdonly
    badinit = 'badinit' # run badinit
    oracle_emdonly = 'oracle_emdonly' # run emdonly
    oracle_lang = 'oracle_lang' # run badinit


    debug = 'debug'
    debug_tool = 'debug_tool'
    debug_phys = 'debug_phys'

    visiononly = 'visiononly' # run visiononly
    cpdeform = 'cpdeform' # run cpdeform

    debug_emdonly = 'debug_emdonly' # run emdonly


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def run_diffphys(task_cfg: DefaultConfig, task: Task):
    date = get_date()

    _sub_cfg = OmegaConf.load(os.path.join(FILEPATH, 'task_cfgs', f'{str(task.value)}.yaml'))
    task_cfg.tool_sampler.n_samples = 1000
    task_cfg = cast(DefaultConfig, OmegaConf.merge(task_cfg, _sub_cfg))

    if task_cfg.saver.use_wandb:
        task_cfg.saver.wandb_group = task_cfg.saver.path.split('/')[-1] + '_' + task.value
        task_cfg.saver.wandb_name = date

    OmegaConf.resolve(cast(DictConfig, task_cfg))
    task_cfg.saver.path = os.path.join(task_cfg.saver.path, date) #TODO: add seed
    from diffsolver.main import main as _main
    return _main, task_cfg

    
def run_rl(task_cfg: DefaultConfig, task: Task):
    from diffsolver.rl_baselines.train_rl import MainConfig, main as _main

    _main_cfg: DictConfig = OmegaConf.structured(MainConfig)

    _sub_cfg = OmegaConf.load(os.path.join(FILEPATH, 'task_cfgs', f'{str(task.value)}.yaml'))
    _main_cfg.merge_with(_sub_cfg)

    # create the gym env
    from diffsolver.rl_baselines.rl_envs import make
    prog = """obj0 = get_iobj('all')
goal = get_goal('all')
tand(
    tkeep(touch(obj0, 0.02), weight=1.),
    tlast(emd(obj0, goal, 0.02), weight=5.)
)"""
    env = make(
        scene=task_cfg.scene,
        max_eipisode_steps=task_cfg.prog.horizon,
        obs_mode='pcd',
        evaluator=dict(metrics =['iou', 'save_scene', 'code']),
        prog=prog
    )


    OmegaConf.resolve(cast(DictConfig, _main_cfg))
    rl_cfg = cast(MainConfig, _main_cfg)

    rl_cfg.path = os.path.join(task_cfg.saver.path, get_date()) #TODO: add seed
    rl_cfg.group_name = task_cfg.saver.path.split('/')[-1] + '_' + task.value
    return lambda: _main(env), rl_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=Task, default=None)
    parser.add_argument('--config', type=str, default=None, required=True)
    args, unknown = parser.parse_known_args()

    
    _task_cfg: DictConfig = OmegaConf.structured(DefaultConfig)
    _task_cfg.merge_with(OmegaConf.load(args.config))
    task_cfg = cast(DefaultConfig, _task_cfg)


    task: Task = args.task
    paths = ['single_stage', task.value, task_cfg.saver.path]
    if 'MODEL_DIR' in os.environ:
        paths = [os.environ['MODEL_DIR']] + paths
    task_cfg.saver.path = os.path.join(*paths)

    if task in [Task.sac, Task.ppo]:
        _main, run_cfg = run_rl(task_cfg, task)
    else:
        _main, run_cfg = run_diffphys(task_cfg, task)


    with tempfile.NamedTemporaryFile(suffix='.yaml') as temp:
        temp.write(OmegaConf.to_yaml(run_cfg).encode('utf-8'))
        temp.flush()

        sys.argv[1:] = ['--config', temp.name] + unknown

        try: 
            import torch
            with torch.device('cuda'):
                _main()
        except Exception as e:
            raise e



if __name__ == '__main__':
    main()