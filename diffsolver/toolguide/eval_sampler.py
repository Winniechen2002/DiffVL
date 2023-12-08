import argparse
import re
import os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass

from typing import cast
from diffsolver.utils import MultiToolEnv
from diffsolver.config import ToolSamplerConfig, SceneConfig

from diffsolver.program.scenes import load_scene_with_envs
from diffsolver.program import progs # noqa: F401
from diffsolver.program.types import SceneSpec
from diffsolver.toolguide import ToolSampler

from tools.utils import logger
from diffsolver.toolguide.tool_progs import constraints

constraints.VERBOSE = True

@dataclass
class MainConfig:
    path: str = 'tmp'
    scene: SceneConfig = SceneConfig()
    sampler: ToolSamplerConfig = ToolSamplerConfig()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args, unknown = parser.parse_known_args()

    cfg: MainConfig = OmegaConf.structured(MainConfig)
    if args.config is not None:
        cast(DictConfig, cfg).merge_with(OmegaConf.load(args.config))
    cast(DictConfig, cfg).merge_with(OmegaConf.from_dotlist(unknown))
    logger.configure(cfg.path, format_strs=['stdout', 'log', 'csv'])

        
    if len(cfg.sampler.lang) > 0 and cfg.sampler.use_lang:
        from diffsolver.toolguide.prompts.get_prompts import answer
        from diffsolver.toolguide.parse_strings import parse2yaml
        result = answer(cfg.sampler.lang)
        new_config = parse2yaml(result)
        cast(DictConfig, cfg).merge_with(new_config)


    env = MultiToolEnv()
    scene_tuple = load_scene_with_envs(env, cfg.scene)
    tool_sampler = ToolSampler(env, cfg.sampler)
    scene = SceneSpec.from_scene_tuple(env, scene_tuple, tool_sampler.select_tool(scene_tuple))
    assert scene is not None
    sols = tool_sampler.update_scene_tuple(scene)
    env.set_state(scene.state)


    goal_image = scene_tuple.get_goal_image()
    if goal_image is not None:
        logger.savefig('goal.png', goal_image)

    images = []
    for q in sols:
        #img = np.concatenate([img, goal_image], axis=1)
        tool_sampler.compute_sdf(q)
        img = env.render('rgb_array')
        images.append(img)

    images = torch.tensor(np.array(images), dtype=torch.uint8).permute(0, 3, 1, 2)

    from torchvision.utils import make_grid
    grid = make_grid(images, nrow=5)
    ndarr = grid.clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    cv2.imwrite(os.path.join(logger.get_dir(), 'sols.png'), ndarr[:,:,::-1])
    
    
if __name__ == '__main__':
    with torch.device('cuda'):
        main()