import gc
import torch
from diffsolver.utils import MultiToolEnv
from diffsolver.config import SceneConfig
from diffsolver.program.types import SceneSpec

from diffsolver.engine import load_scene_with_envs
from diffsolver.toolguide import ToolSamplerConfig, ToolSampler



def test():
    env = MultiToolEnv()

    state_tuple = load_scene_with_envs(env, SceneConfig())

    config = ToolSamplerConfig(
        sampler='default',
        optimize_loss=False,
        equations=['grasp(0)'],
        constraints=['less(ty(), 0.15)', 'collision_free'],
    )

    tool_sampler = ToolSampler(env, config)

    for _ in range(100):
        with torch.device("cuda:0"):
            _, qpos, _ = tool_sampler.solve(state_tuple, tool_sampler.select_tool(state_tuple))

        state_tuple.state.qpos = qpos.detach().cpu().numpy()
        env.set_state(state_tuple.state)
        obs = env.get_obs()
        qpos2 = obs['qpos']
        assert torch.allclose(qpos, qpos2)
        assert qpos2[1] < 0.15
        assert env.get_obs()['dist'].min() > 0.0

    env.render('test.png')

    del env, tool_sampler
    gc.collect()


if __name__ == '__main__':
    test()