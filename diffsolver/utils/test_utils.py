import matplotlib.pyplot as plt
from diffsolver.utils import MultiToolEnv


def build_wind_env():
    env = MultiToolEnv()
    from diffsolver.utils import load_scene
    scene_tuple = load_scene("wind_gripper.yml", clear_cache=True)
    env.set_state(scene_tuple.state)
    return env, scene_tuple


def generate_scene_traj(T=30):
    from diffsolver.program.types import SceneSpec
    env, scene_tuple = build_wind_env()

    obs = [env.get_obs()]

    assert env.action_space.shape == (7,)
    for _ in range(T):
        env.step(env.action_space.sample())
        obs.append(env.get_obs())

    scene = SceneSpec(env, obs, scene_tuple, None) # type: ignore
    return env, scene