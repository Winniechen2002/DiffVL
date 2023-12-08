import diffsolver.set_render_devices
from diffsolver.utils.renderer import CameraGUI
import cv2
from diffsolver.rl_baselines.rl_envs.gym_env import LookatConfig, MultiToolEnv, CN, SceneConfig, load_scene_with_envs, ToolSampler, ToolSamplerConfig, SceneSpec
from omegaconf import OmegaConf
from typing import Any, Dict, List, cast

def test_ray_tracing():
    lookat = LookatConfig(theta=0.3)
    gui = CameraGUI(offscreen=True, ray_tracing=256)
    gui.lookat(lookat)
    env = MultiToolEnv(sim_cfg=CN(dict(max_steps=100)))

    yaml = """
path: 22_0.task
Tool:
    tool_name: Gripper
    qpos: [0.2730940878391266, 0.058373261243104935, 0.6100195050239563, 1.5707963705062866, 1.5707963705062866, 0.0, 0.03885454311966896]
    size: (0.02, 0.06, 0.02)
    friction: 5.
"""
    cfg = OmegaConf.create(yaml)

    scene = cast(SceneConfig, OmegaConf.merge(OmegaConf.structured(SceneConfig), cfg))
    # scene = SceneConfig()
    scene_tuple = load_scene_with_envs(env, scene)
    tool_sampler = ToolSampler(env, ToolSamplerConfig(n_samples=0))
    scene = SceneSpec.from_scene_tuple(
        env, scene_tuple, tool_sampler.select_tool(scene_tuple)
    )

    env.set_state(scene_tuple.state)
    gui.reset(env)
    img = gui.capture()

    cv2.imwrite('haha.png', img[..., [2,1,0]])

    
if __name__ == '__main__':
    test_ray_tracing()
