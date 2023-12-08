import torch
import gc
import numpy as np
import tqdm
from omegaconf import OmegaConf

from diffsolver.utils import MultiToolEnv
from diffsolver.config import ToolSamplerConfig, SceneConfig
from diffsolver.program.scenes import load_scene_with_envs
from diffsolver.toolguide import ToolSampler
from diffsolver.utils.sdf import compute_sdf


def test():
    with torch.device('cuda'):
        env = MultiToolEnv()
        tool_cfg = OmegaConf.structured(ToolSamplerConfig)

        scene_cfg = OmegaConf.structured(SceneConfig)

        for i in tqdm.trange(10):
            qpos = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.], dtype=torch.float32)
            qpos[:3] = torch.rand(size=(3,), dtype=torch.float32)
            qpos[3:6] = torch.rand(size=(3,), dtype=torch.float32) * 2 * 3.1415
            qpos[6] = torch.rand(size=(1,), dtype=torch.float32) * 0.1
            state = load_scene_with_envs(env, scene_cfg)

            state.state.tool_cfg.defrost() # type: ignore
            if np.random.rand() < 0.5:
                state.state.tool_cfg.mode = 'Capsule' # type: ignore
            else:
                state.state.tool_cfg.mode = 'Box' # type: ignore
            state.state.tool_cfg.size = (np.random.rand(3) * 0.1 + 0.01).tolist() # type: ignore
                
            env.set_state(state.state)
            assert env.tool_cur is not None

            #qpos = env.get_obs()['qpos']

            tools = ToolSampler(env, tool_cfg) 
            dist = tools.compute_sdf(qpos)
            dist2 = compute_sdf(env, qpos, torch.tensor(state.state.X, dtype=torch.float32))

            assert torch.allclose(dist, dist2, atol=1e-6), f"{torch.abs(dist - dist2).max()}"

        del env, state, tools # type: ignore
        gc.collect()
    

if __name__ == '__main__':
    for j in range(100):
        test()