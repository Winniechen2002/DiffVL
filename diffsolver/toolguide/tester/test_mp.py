import gc
import numpy as np
import torch
from diffsolver.utils import MultiToolEnv
from diffsolver.config import SceneConfig, MPConfig
from diffsolver.program.types import SceneSpec

from diffsolver.engine import load_scene_with_envs
from diffsolver.toolguide import ToolSamplerConfig, ToolSampler
from tools.utils import logger




def test():
    env = MultiToolEnv()
    logger.configure('tmp')

    state_tuple = load_scene_with_envs(env, SceneConfig())
    state_tuple.state.X[:, 0] -= 0.3
    env.set_state(state_tuple.state)
    start = env.render('rgb_array')

    config = ToolSamplerConfig(
        sampler='default',
        optimize_loss=False,
        equations=['vgrasp(0)'],
        constraints=['less(ty(), 0.15)', 'collision_free'],
        motion_planner=MPConfig(max_iter=10000, info=True),
    )

    tool_sampler = ToolSampler(env, config)
    scene = SceneSpec.from_scene_tuple(env, state_tuple, tool_sampler.select_tool(state_tuple), requires_grad=False)

    tool_sampler.update_scene_tuple(scene)
    env.set_state(scene.state_tuple.state)
    end = env.render('rgb_array')

    import matplotlib.pyplot as plt
    assert start is not None and end is not None and start.shape == end.shape
    plt.imshow(np.concatenate([start, end], axis=1))
    plt.savefig('test.png')

    del env, tool_sampler, scene, config
    gc.collect()


if __name__ == '__main__':
    test()