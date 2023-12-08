import diffsolver.set_render_devices
import tqdm
import sys
from diffsolver.utils.renderer import CameraGUI
import torch
import cv2
from diffsolver.rl_baselines.rl_envs.gym_env import LookatConfig, MultiToolEnv, CN, SceneConfig, load_scene_with_envs, ToolSampler, ToolSamplerConfig, SceneSpec
from omegaconf import OmegaConf
from typing import Any, Dict, List, cast


def render(path):
    lookat = LookatConfig(theta=0.3)
    gui = CameraGUI(offscreen=True, ray_tracing=256)
    gui.lookat(lookat)
    env = MultiToolEnv(sim_cfg=CN(dict(max_steps=100)))

    trajs = torch.load(path)
    images = []
    for state in tqdm.tqdm(trajs['states'], total=len(trajs['states']), desc='rendering'):
        env.set_state(state)
        gui.reset(env)
        img = gui.capture()
        images.append(img)


    return images

    
if __name__ == '__main__':
    from tools.utils import animate
    images = render(sys.argv[1])
    animate(images, sys.argv[2])