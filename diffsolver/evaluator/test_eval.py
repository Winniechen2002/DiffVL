import gym
from diffsolver import set_render_devices
from diffsolver.rl_baselines.rl_envs import make
from tools.utils import animate


def run():
    env = gym.make('Lift-v0', evaluator=dict(metrics=['iou', 'save_scene', 'code']))

    images = []
    env.reset()
    images.append(env.render(mode='rgb_array'))

    while True:
        action = env.action_space.sample()
        _, _, done, info = env.step(action)
        images.append(env.render(mode='rgb_array'))
        if done:
            break

    print(info)
    animate(images)

if __name__ == '__main__':
    run()