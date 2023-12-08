import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import cast
from diffsolver.utils import MultiToolEnv
from diffsolver.config import SceneConfig
from diffsolver.program.types import SceneSpec
from omegaconf import OmegaConf

from diffsolver.engine import load_scene_with_envs
from diffsolver.toolguide import ToolSamplerConfig, ToolSampler
from tools.utils import logger
from diffsolver.toolguide.tool_progs import constraints

constraints.VERBOSE = True


scene="""
path: 21_0.task
Tool:
#     tool_name: Pusher
#     qpos: [0.5, 0.4, 0.5, 0., 0., 0.]
#     size: (0.02, 0.15, 0.15)
#     friction: 0.
    tool_name: DoublePushers
    qpos: [0.3, 0.1, 0.5, 0., 0., 0., 0.7, 0.1, 0.5, 0., 0., 0.]
    size: (0.02, 0.05, 0.02)
"""



def run():
    logger.configure('tmp')

    env = MultiToolEnv()

    scene_config = OmegaConf.merge(OmegaConf.structured(SceneConfig), OmegaConf.create(scene))

    state_tuple = load_scene_with_envs(env, cast(SceneConfig, scene_config))

    goal_image = state_tuple.get_goal_image()
    assert goal_image is not None
    env.set_state(state_tuple.state)
    img = env.render('rgb_array')
    plt.imshow(np.concatenate([img, goal_image], axis=1))
    plt.savefig('scene_goal.png')

    config = ToolSamplerConfig(
        sampler='default',
        n_samples=10000,
        optimize_loss=True,
        equations=['vertical'],
        #constraints=['less(ty(), 0.15)', 'collision_free'],
        constraints=['collision_free', 'cpdeform(0)'],
    )

    tool_sampler = ToolSampler(env, config)
    _, qpos = tool_sampler.solve(state_tuple, tool_sampler.select_tool(state_tuple))

    print(qpos)
    state_tuple.state.qpos = qpos.detach().cpu().numpy()
    env.set_state(state_tuple.state)

    env.render('test.png')

    del env, tool_sampler, state_tuple
    gc.collect()


if __name__ == '__main__':
    with torch.device("cuda:0"):
        run()