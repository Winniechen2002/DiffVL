from envs.plb import MultiToolEnv
from tools.utils import animate
import tqdm
from frontend import DATA_PATH

quality = 2
# from envs import soft_utils
# soft_utils.N_VOLUME = 40000
# print(env.simulator.dt)

import numpy as np
import os
import pickle

from envs.world_state import WorldState
from envs.test_utils import init_scene
from llm.tiny import Scene

def test():
    env = MultiToolEnv()
    init_scene(env, 0)
    # state.E_nu_yield[:, 0] = 100
    # state.E_nu_yield[:, 2] = 30

    pkl_path = os.path.join(DATA_PATH, 'task_{}'.format(10))
    path = os.path.join(pkl_path, 'scene_{}.pkl'.format(1))
    with open(path, 'rb') as f:
        print('load scene from', path)
        states = pickle.load(f)
    env.set_state(states.state)

    state: WorldState = env.get_state()

    state = state.switch_tools('Gripper', np.array([0.7, 0.07, 0.7, 0., 0., np.pi/2, 0.05]), friction=10., size=(0.02, 0.15, 0.02), action_scale=(0.015, 0.015, 0.015, 0., 0., 0.0, 0.015), softness=0.)
    images = []
    env.set_state(state)

    for i in tqdm.trange(270):
        if i < 75:
            action = [0., 0.2, 0., 0., 0., 0., 0. ]
        elif i < 215:
            action = [0., 0., -0.2, 0., 0., 0., 0. ]
        else:
            action = [0., -0.2, 0., 0., 0., 0., 0. ]
        env.step(action)
        images.append(env.render('rgb_array', spp=1))


    animate(images)