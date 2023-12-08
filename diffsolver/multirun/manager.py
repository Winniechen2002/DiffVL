import hashlib
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
from typing import List, cast, MutableMapping, Mapping

from diffsolver.multirun.multirun_config import ManagerConfig, BaseConfigType, SubConfig, Tasks
from diffsolver.multirun.renderer import HTMLRenderer


class Manager:
    def __init__(self, config: ManagerConfig | DictConfig, config_path: str) -> None:
        self.config = config

        base = OmegaConf.structured(DefaultConfig)
        self.base = base
        if self.config.base_path is not None:
            self.base_config = OmegaConf.merge(base, OmegaConf.load(self.config.base_path))
            base_dir = os.path.dirname(self.config.base_path)
        else:
            self.base_config = None
            base_dir = os.path.dirname(config_path)

        self.base_dir = base_dir # used to load the sub configs
        # self.exp = cast(DefaultConfig, OmegaConf.merge(base, self.config.common))
        #self.common = OmegaConf.merge(base, self.config.common)


    def build_sweeper(self, date=None) -> Mapping[str, BaseConfigType]:
        date = date or datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        root_path = os.path.join(self.config.path, date)

        if self.config.sweep_mode == 'list':
            variations = self.config.variations
            if len(variations) == 0:
                variations = [SubConfig()] # append an empty config so that we can just run the base config for mutliple times 

            configs: MutableMapping[str, BaseConfigType] = {}
            subconfigs: MutableMapping[str, BaseConfigType] = {}

            for idx, variation in enumerate(variations):
                if variation.config is not None:
                    base_config = OmegaConf.load(os.path.join(self.base_dir, variation.config)) 
                    base_config = OmegaConf.merge(self.base, base_config)
                else:
                    base_config = self.base_config
                base_config = OmegaConf.merge(base_config, self.config.common)
                modifier = variation.modifier
                merged = OmegaConf.merge(base_config, modifier)

                assert isinstance(merged, DictConfig), 'sub_config should be a DictConfig'

                if variation.name is None:
                    variation.name = f'{idx}'
                sub_config: BaseConfigType = cast(DefaultConfig, merged) # this is just a duck typing ...
                subconfigs[variation.name] = sub_config


            seeds = self.config.seeds
            if seeds is None:
                seeds = [None]

            for seed in seeds:
                last_saving_path: str | None = None
                for name, sub_config in subconfigs.items():
                    if seed is not None:
                        name = f'{name}_seed_{seed}' # group by seeds
                    path = os.path.join(root_path, name)
                    _sub_config = copy.deepcopy(sub_config)

                    # set the path and seed
                    _sub_config.saver.path = path
                    _sub_config.seed = seed
                    _sub_config.saver.logger.date = False

                    if self.config.subsequent and last_saving_path is not None:
                        _sub_config.scene.use_config = True
                        _sub_config.scene.path = os.path.join(last_saving_path, 'trajs.pt')

                    last_saving_path = path

                    configs[name] = _sub_config
            return configs
        else:
            raise NotImplementedError

    def render(self, path: str|None=None):
        self.sub_configs = self.build_sweeper(path)

        for name, sub_config in self.sub_configs.items():
            print(termcolor.colored(f'Rendering {name}', 'green'))
            print(OmegaConf.to_yaml(sub_config))

        #path = path or self.config.path
        if path is not None:
            path = os.path.join(self.config.path, path)
        else:
            path = self.config.path
        renderer = HTMLRenderer(self.config.renderer, path, self.sub_configs)
        renderer.run()

    def run(self):
        self.sub_configs = self.build_sweeper()

        # write the config to the path
        config_root = tempfile.gettempdir()

        # get hash from self.config
        config_hash = hashlib.md5(OmegaConf.to_yaml(self.config).encode('utf-8')).hexdigest()
        config_root = os.path.join(config_root, 'multirun_config', config_hash, datetime.datetime.now().strftime(f"%Y-%m-%d-%H-%M-%S-%f"))
        os.makedirs(config_root, exist_ok=True)

        paths = []
        for name, sub_config in self.sub_configs.items():
            match = re.match(r'(.*)_seed_\d+', name)
            if match is not None:
                base_name = match.group(1)
            else:
                base_name = name

            if self.config.subset is not None and base_name not in self.config.subset:
                continue

            path = os.path.join(config_root, name + '.yaml')
            paths.append(path)
            OmegaConf.save(sub_config, path)

        OmegaConf.save(self.config, os.path.join(config_root, 'manager.yaml'))
        self.run_configs(paths, self.config.max_retry)

        
    def run_configs(self, paths: List[str], max_retry):
        #TODO: use multiprocessing and multiple GPUS to run the configs
        for path in paths:
            print(f'python -m diffsolver.main --config {path}')
            if not self.config.debug_run:
                retry = max_retry
                while retry:
                    ret = os.system(f'python -m diffsolver.main --config {path}')
                    if ret != 0:
                        retry -= 1
                        print('RuntimeError(Error running {})'.format(path))
                    else:
                        retry = 0


def main(default_task=Tasks.render):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    args, unknown = parser.parse_known_args()

    default_conf: DictConfig = OmegaConf.structured(ManagerConfig)
    default_conf.task = default_task
    if args.config is not None:
        default_conf.merge_with(OmegaConf.load(args.config))

    input_cfg = OmegaConf.from_dotlist(unknown)
    default_conf.merge_with(input_cfg)
    print(OmegaConf.to_yaml(default_conf))

    manager = Manager(default_conf, args.config)

    if manager.config.task == Tasks.run:
        manager.run()
    else:
        manager.render(args.path)
    
    
if __name__ == '__main__':
    main()