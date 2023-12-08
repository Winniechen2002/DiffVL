import argparse
import gc
import torch
from omegaconf import OmegaConf, DictConfig
from diffsolver.utils import MultiToolEnv
from diffsolver.engine import Engine
from diffsolver.config import DefaultConfig

ENV = None # use global variable to reuse the env for different run

def build_engine_from_config() -> Engine:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args, unknown = parser.parse_known_args()

    default_conf: DictConfig = OmegaConf.structured(DefaultConfig)
    if args.config is not None:
        default_conf.merge_with(OmegaConf.load(args.config))

    input_cfg = OmegaConf.from_dotlist(unknown)
    default_conf.merge_with(input_cfg)
    print(OmegaConf.to_yaml(default_conf))

    global ENV

    if ENV is None:
        ENV = MultiToolEnv(
            sim_cfg=dict(max_steps=default_conf.max_steps)  # type: ignore
        )
    engine = Engine(ENV, default_conf)
    return engine


def main():
    engine = build_engine_from_config()
    engine.main()
    del engine
    gc.collect()

if __name__ == '__main__':
    with torch.device('cuda:0'):
        main()