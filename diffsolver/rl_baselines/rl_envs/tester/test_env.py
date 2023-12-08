from diffsolver import set_render_devices
import gym
import gc
from tools.utils import animate
from diffsolver.rl_baselines.rl_envs import make 

def run_env():
    env = gym.make('Lift-v0', obs_mode='pcd')
    print(env.observation_space)
    print(env.action_space)

    obs = env.reset()
    assert obs in env.observation_space
    images = []
    while True:
        o, r, done, _ = env.step(env.action_space.sample())
        if done:
            break
        images.append(env.render(mode = 'rgb_array'))

    animate(images)

    del env
    gc.collect()

if __name__ == '__main__':
    run_env()