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
    import torch
    from diffsolver.program.scenes.visiontask import TaskSeq

    path = 'multistage/mid/task48/20230516053121636478/'

    gui = CameraGUI(offscreen=True, ray_tracing=256)
    lookat = LookatConfig()
    gui.lookat(lookat)
    gui.setup_camera(pose = [0.5, 0.6, 1.5])
    env = MultiToolEnv(sim_cfg=CN(dict(max_steps=100)))

    for i in range(5):

        stage_path = os.path.join(path, 'stage_{}'.format(i))

        data_frame = torch.load(os.path.join(stage_path, 'trajs.pt'))
        states: Sequence[WorldState] = data_frame['states']

        final_state = states[len(states) - 1]
        final_state.qpos = final_state.get_empty_state_qpos()
        final_state.qpos[0] = 0
        final_state.qpos[1] = 0

        env.set_state(final_state)
        gui.reset(env)
        img = gui.capture()

        cv2.imwrite('task48_mid_result_{}.png'.format(i), img[..., [2,1,0]])

def test_ray_tracing_goal():
    import gc
    import cv2
    import os
    import matplotlib.pyplot as plt
    from diffsolver.program.scenes.visiontask import TaskSeq

    task_id = 48
    gui = CameraGUI(offscreen=True, ray_tracing=256)
    lookat = LookatConfig(theta=0.3)
    gui.lookat(lookat)
    gui.setup_camera(pose = [0.5, 0.6, 1.7])
    env = MultiToolEnv(sim_cfg=CN(dict(max_steps=100)))
    task = TaskSeq(task_id)

    for i in range(20):
        state, _ = task.fetch_stage(i)

        env.set_state(state)
        gui.reset(env)
        img = gui.capture()

        cv2.imwrite('{}_{}.png'.format(task_id,i), img[..., [2,1,0]])

if __name__ == '__main__':
    test_ray_tracing()