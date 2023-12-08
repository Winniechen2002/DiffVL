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
    from diffsolver.program.scenes.visiontask import TaskSeq

    # with open('program/scenes/visiontask/task_id.txt', 'r') as file:
        # task_ids = [int(line.strip()) for line in file]

    for task_id in [8,22,57]:    
        gui = CameraGUI(offscreen=True, ray_tracing=256)
        lookat = LookatConfig(theta = 0, phi = 3.14/3, zeta = 3.14/2, radius = 1.5)
        gui.lookat(lookat)
        env = MultiToolEnv(sim_cfg=CN(dict(max_steps=100)))
        path = 'figs/phi{}theta{}zeta{}r{}'.format(lookat.phi, lookat.theta, lookat.zeta, lookat.radius)
        if not os.path.exists(path):
            os.mkdir(path)
        try:
            task = TaskSeq(task_id)

            for i in range(20):
                try:
                    state, _ = task.fetch_stage(i)

                    state.qpos[0] = -1
                    state.qpos[1] = -1
                    env.set_state(state)
                    gui.reset(env)
                    img = gui.capture()

                    cv2.imwrite(path + '/{}_{}.png'.format(task_id,i), img[..., [2,1,0]])
                    # print(path + '/{}_{}.png'.format(task_id,i), img[..., [2,1,0]])
                    del state, _, img
                except Exception:
                    print('We do not have scene{}'.format(i))
            del task
        except Exception:
            print('We do not have task{}'.format(task_id))
        del gui, lookat, env
        gc.collect()

    
if __name__ == '__main__':
    test_ray_tracing()