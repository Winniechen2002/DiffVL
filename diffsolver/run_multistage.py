import sys
import re
import copy
from enum import Enum
import termcolor
import os
from omegaconf import OmegaConf, DictConfig
import argparse
from diffsolver.config import DefaultConfig
import datetime
import tempfile
from typing import List, cast, MutableMapping, Mapping, Any, List, Mapping, Union, Tuple
from dataclasses import dataclass, field
from tools.utils import logger

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class Task(Enum):
    prog = 'prog' # run the program-based multi-stage solver
    dev = 'dev'
    lang = 'lang'
    finalonly = 'finalonly'
    noresample = 'noresample'
    single = 'single'
    long_horizon = 'long'
    short_horizon = 'short'
    mid_horizon = 'mid'
    test = 'test'
    sac = 'sac'
    ppo = 'ppo'


@dataclass
class SubConfig:
    config: str
    modifier: Any = field(default_factory=lambda: {})

@dataclass
class ManagerConfig:
    path: str                                           # path to store all runs
    base_path: str                                      # path to the base config
    final_goal: str                                     # the final goal to achieve

    max_retry: int = 3
    subsequent: bool = False

    common: Any = field(default_factory=lambda: {})     # modifier over the base config

    stages: List[SubConfig] = field(default_factory=list)   # a list of variations to sweep over the base config, used when sweep_mode is list


    skip: str|None = None                                   # skip this run, this is used for debugging: if skip is not None, we can directly start from a certain stage instead of running from the beginning
    skip_stages: List[int] = field(default_factory=list)    # skip some stages


    onlyfinal: bool = False                                 # only optimize for the final goal
    resample_tool: bool = True                              # resample tool for each stage
    single_stage: bool = False                              # only run one stage

    rl_method: str|None = None                                     # is rl task


class Multirun:
    def __init__(self, task: Task, _config:DictConfig, dotlist, config_path: str) -> None:
        """
        NOTE: task_sub_cfg has a hight priority (except for the modifier, which is the highest). 
        Its common part will be used to override all subconfigs
        """
        self.task = task
        self.base_dir = os.path.dirname(config_path)
        assert not OmegaConf.is_missing(_config, 'final_goal'), 'the final goal should be specfiied in the config'


        self._task_sub_cfg = OmegaConf.load(os.path.join(FILEPATH, 'task_cfgs', f'multistage_{str(task.value)}.yaml'))
        self._task_sub_cfg.merge_with(OmegaConf.from_dotlist(dotlist))

        assert isinstance(self._task_sub_cfg, DictConfig), 'config should be a DictConfig'
        self.config = cast(ManagerConfig, OmegaConf.merge(_config, self._task_sub_cfg))


        base_config: DictConfig = OmegaConf.structured(DefaultConfig)

        if not OmegaConf.is_missing(self.config, 'base_path'):
            base_config.merge_with(self.load_config(self.config.base_path))
        base_config.merge_with(OmegaConf.create(self.config.common))
        self.base_config = cast(DefaultConfig, base_config)

        self.skip_path = None



    def load_config(self, config_path: str) -> DictConfig:
        out = OmegaConf.load(os.path.join(self.base_dir, config_path))
        assert isinstance(out, DictConfig), 'config should be a DictConfig'
        return out


    def build_configs(self, saver = None)-> Tuple[List[DefaultConfig], str, str, str]:
        date = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

        if saver:
            paths = ['multistage', self.task.value, saver, self.config.path, date]
        else:
            paths = ['multistage', self.task.value, self.config.path, date] 
        if 'MODEL_DIR' in os.environ:
            paths = [os.environ['MODEL_DIR']] + paths


        path = os.path.join(*paths)

        if self.config.skip is not None:
            self.skip_path = os.path.join(*paths[:-1], self.config.skip)


        wandb_group = f'{self.task.value}_{self.config.path}'
        name = date


        variations = self.config.stages

        stages = []
        last_saving_path = ''

        for idx, variation in enumerate(variations):
            _cfg = OmegaConf.merge(self.base_config, self.load_config(variation.config))
            _cfg = OmegaConf.merge(_cfg, self.config.common, variation.modifier)
            cfg = cast(DefaultConfig, _cfg)

            if idx > 0:
                cfg.scene.path = os.path.join(last_saving_path, 'trajs.pt')
                cfg.scene.use_config = True

                if not self.config.resample_tool:
                    cfg.tool_sampler.n_samples = 0
                    cfg.scene.Tool = {}

            last_saving_path = cfg.saver.path = os.path.join(path, f'stage_{idx}')

            if self.config.onlyfinal:
                # ablation study
                cfg.scene.goal = self.config.final_goal
                cfg.prog.translator.version = 'emdonly'
                cfg.prog.max_retry = 1


            stages.append(cfg)

            if self.config.single_stage:
                break

        return stages, path, wandb_group, name

    def run_rl(self):
        #from diffsolver.rl_baselines.run_ import main as _main
        from diffsolver.run_single_stage import run_rl, Task
        self.config.single_stage = True
        task = Task(self.config.rl_method)
        config = self.build_configs()[0][0]
        # get the single stage
        config.saver.path = os.path.abspath(os.path.join(config.saver.path, '../..'))
        _main, rl_config = run_rl(config, task)
        assert rl_config.group_name is not None
        rl_config.group_name = rl_config.group_name + '_multi'


        with tempfile.NamedTemporaryFile(suffix='.yaml') as temp:
            temp.write(OmegaConf.to_yaml(rl_config).encode('utf-8'))
            temp.flush()
            sys.argv[1:] = ['--config', temp.name]
            _main()




    def run(self, saver = None):
        if self.config.rl_method is not None:
            return self.run_rl()
        
        sub_configs, save_path, wandb_group, name = self.build_configs(saver = saver)


        # write the config to the path
        config_root = tempfile.gettempdir()

        # get hash from self.config
        import hashlib
        config_hash = hashlib.md5(OmegaConf.to_yaml(self.config).encode('utf-8')).hexdigest()
        config_root = os.path.join(config_root, 'multirun_config', config_hash, datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
        os.makedirs(config_root, exist_ok=True)

        paths = []
        for idx, sub_config in enumerate(sub_configs):
            path = os.path.join(config_root, f'stage_{idx}.yaml')
            paths.append(path)
            OmegaConf.save(sub_config, path)

        saver = self.base_config.saver
        format_strs = saver.logger.format_strs
        kwargs = {}
        if saver.use_wandb:
            kwargs = {'project': os.environ.get("WANDB_PROJECT_NAME", 'diffsolver'), 'group': 'multi_'+wandb_group, 'name': name, 'config': self.config}
            format_strs += ['wandb']
        logger.configure(dir=save_path, format_strs=format_strs, date=False, **kwargs)

        import torch
        from diffsolver.evaluator import evaluator
        evaluator.FINAL_GOAL = self.config.final_goal

        with torch.device('cuda'):
            for stage_id, path in enumerate(paths):
                if self.skip_path is not None and stage_id in self.config.skip_stages:
                    cmd = f'cp -r {self.skip_path}/stage_{stage_id} {sub_configs[stage_id].saver.path}'
                    error = os.system(cmd)
                    assert error == 0, 'copying failed when executing: '+cmd
                else:
                    from diffsolver.main import main as _main
                    sys.argv[1:] = ['--config', path]
                    _main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=Task, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--saver', type=str, default=None)
    args, unknown = parser.parse_known_args()

    default_conf: DictConfig = OmegaConf.structured(ManagerConfig)
    if args.config is not None:
        default_conf.merge_with(OmegaConf.load(args.config))

    manager = Multirun(args.task, default_conf, unknown, args.config)
    manager.run(saver = args.saver)
    
    
if __name__ == '__main__':
    main()

