# %%
import diffsolver.set_render_devices
from diffsolver.utils.renderer import CameraGUI
import cv2
from diffsolver.rl_baselines.rl_envs.gym_env import LookatConfig, MultiToolEnv, CN, SceneConfig, load_scene_with_envs, ToolSampler, ToolSamplerConfig, SceneSpec
from omegaconf import OmegaConf
from typing import Any, Dict, List, cast

def test_ray_tracing():
    import gc
    import cv2
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import torch
    from PIL import Image
    from diffsolver.program.scenes.visiontask import TaskSeq

    # with open('program/scenes/visiontask/task_id.txt', 'r') as file:
        # task_ids = [int(line.strip()) for line in file]
    task_id = 9  
    gui = CameraGUI(offscreen=True, ray_tracing=256)
    lookat = LookatConfig(theta = 0, phi = 3.14/4, zeta = 3.14/2, radius = 2)
    gui.lookat(lookat)
    env = MultiToolEnv(sim_cfg=CN(dict(max_steps=100)))

    task = TaskSeq(task_id)

    state, _ = task.fetch_stage(1)


    state.qpos[0] = -1
    state.qpos[1] = -1

    color_dict = [(187,255,255), (187,255,255), (255,187,255), (255,255,224), (255,174,185), (255,187,255)]

    imgs = []
    for i in range(1,5):
        if i == 1:
            from envs.soft_utils import sphere
            pcd = torch.tensor(sphere(0.1, center=(0.5, 0.1, 0.3), n = None)).cuda().float()
            
            colors = np.zeros((len(pcd), 3))
            colors[:,:] = color_dict[i]
        else:
            f = open('program/scenes/visiontask/task_8_objects/scene_{}.pkl'.format(i+1), 'rb')
            state_obj = pickle.load(f).state
            object_id = np.unique(state_obj.ids)
            object_dict = {int(i): np.where(state_obj.ids ==j)[0] for i, j in zip(object_id, object_id)}
            pcd = torch.tensor(state_obj.X)[object_dict[1]].cuda()
            colors = np.zeros((len(pcd), 3))
            colors[:,:] = color_dict[i]

        env.set_state(state)
        gui.reset(env)

        img = gui.capture()
        gui._add_pcd(pcd, colors, transmission = 0)

        tmp = gui.capture()
    
        imgs.append(img)
        img = img * 0.5 + tmp * 0.5
        cv2.imwrite('task8/{}_{}.png'.format(task_id,i), img[..., [2,1,0]])

    out = None
    for img in imgs:
        if out is None:
            out = 0.3*img
        else:
            out = out + 0.3*img
    cv2.imwrite('task8/{}.png'.format(task_id), out[..., [2,1,0]])

    env.set_state(state)
    gui.reset(env)

    img = gui.capture()
    cv2.imwrite('task8/stage_1.png', img[..., [2,1,0]])

    state, _ = task.fetch_stage(2)

    env.set_state(state)
    gui.reset(env)

    img = gui.capture()
    cv2.imwrite('task8/stage_2.png', img[..., [2,1,0]])
   
if __name__ == '__main__':
    test_ray_tracing()