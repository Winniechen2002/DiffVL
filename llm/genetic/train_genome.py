#!/usr/bin/env python3
import os
import glob
import yaml
import argparse

from llm.genetic.loader import load_prog

from llm.genetic.utils import GENOMEPATH, SOLPATH
from llm.genetic.trainer import Trainer
from llm.genetic.frankwolfe import FrankWolfe


def main():
    tasks = [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob(f"{GENOMEPATH}/*.py"))]

    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=tasks)
    parser.add_argument("--show_stage_info", action='store_true')
    parser.add_argument("--ignore_constraint", action='store_true')

    parser.add_argument("--all", action='store_true')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--replay", action='store_true')
    args, _ = parser.parse_known_args()


    prog = load_prog(os.path.join(GENOMEPATH, args.task+'.py'))[0]

    if prog.doc is not None:
        optim_cfg = yaml.load(prog.doc, Loader=yaml.FullLoader)
    else:
        optim_cfg = {}

    if not args.ignore_constraint:
        TRAINER = FrankWolfe
    else:
        TRAINER = Trainer
        # hack here, remove the configs for constrained optimizer ..

        if 'reg_proj' in optim_cfg:
            optim_cfg.pop('reg_proj')

        if 'weight_penalty' in optim_cfg:
            optim_cfg.pop('weight_penalty')

    if 'path' not in optim_cfg:
        optim_cfg['path']=os.path.join(SOLPATH, args.task)
    optim_cfg['n_stages'] = 2

    trainer = TRAINER.parse(prog, env=None, parser=parser, use_wandb=args.wandb, **optim_cfg)

    # hack here ..
    T = trainer.get_stage_info()
    if args.show_stage_info:
        trainer.env.render('xx.png')
        exit(0)

    actions = trainer.get_initial_action(trainer._cfg.T or T)
    cfg = trainer._cfg

    if args.replay:
        n_stage = len(trainer.stage_timesteps)
        actions = trainer.get_initial_action(trainer._cfg.T or T, cfg.path + f"/{n_stage}_{cfg.n_stages}")
        from tools.utils import logger
        logger.configure('tmp')
        trainer.write_video(actions, filename='replay')
        exit(0)

    if cfg.end_stage is None:
        assert args.all

        for i in range(len(trainer.stage_timesteps)):
            actions = trainer.main_loop(actions, end_stage=i+1)
    else:
        assert cfg.end_stage is not None

        if cfg.end_stage > 1 and cfg.init_actions is None:
            actions = trainer.get_initial_action(trainer._cfg.T or T, cfg.path + f"/{cfg.end_stage-1}_{cfg.n_stages}")

        for i in range(cfg.end_stage-1, len(trainer.stage_timesteps)):
            actions = trainer.main_loop(actions, end_stage=i+1)
            if not args.all:
                break

    
if __name__ == '__main__':
    main()