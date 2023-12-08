import matplotlib.pyplot as plt
import tqdm
from diffsolver.config import DefaultConfig
from tools.utils import animate 
from omegaconf import OmegaConf, DictConfig
from diffsolver.engine import load_scene_with_envs, MultiToolEnv


scene="""
scene:
    path: block_to_cut.yml
    Tool:
        tool_name: Pusher
        qpos: [0.5, 0.4, 0.5, 0., 0., 0.]
        size: (0.02, 0.15, 0.15)
        friction: 0.
"""

def main():
    config = OmegaConf.structured(DefaultConfig)
    config = OmegaConf.merge(config, OmegaConf.create(scene))

    env = MultiToolEnv()
    scene_tuple = load_scene_with_envs(env, config.scene)

    env.set_state(scene_tuple.state, requires_grad=False)

    # image = env.render('rgb_array')
    # assert image is not None
    print(env.action_space)

    images = []

    for i in tqdm.trange(100):
        actions = []
        if i < 60:
            actions = [0., -0.3, 0., 0., 0., 0.]
        else:
            actions = [1., 0., 0., 0., 0., 0.]

        obs, reward, done, info = env.step(actions)

        images.append(env.render('rgb_array'))

    animate(images)

    
if __name__ == '__main__':
    main()