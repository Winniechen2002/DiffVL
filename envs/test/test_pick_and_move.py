from envs.plb import MultiToolEnv
from tools.utils import animate
import tqdm

quality = 2
# from envs import soft_utils
# soft_utils.N_VOLUME = 40000
# print(env.simulator.dt)

import numpy as np

from envs.world_state import WorldState
from envs.test_utils import init_scene

def test():
    env = MultiToolEnv(sim_cfg=dict(ground_friction=1., yield_stress=200., dx=1./64/quality, dt=0.00005/quality, substeps=20 * 2 * quality, vol=(1./64/quality/2)**3, mass=(1./64/quality/2)**3, n_particles=20000))
    init_scene(env, 0)
    state: WorldState = env.get_state()
    # state.E_nu_yield[:, 0] = 100
    # state.E_nu_yield[:, 2] = 30

    state = state.switch_tools('Gripper', np.array([0.5, 0.15, 0.5, 0., 0., 0., 0.9]), friction=10., size=(0.02, 0.15, 0.02), action_scale=(0.015, 0.015, 0.015, 0., 0., 0.0, 0.015), softness=0.)
    images = []
    env.set_state(state)

    for i in tqdm.trange(100):
        env.step([0., 0.2, 0., 0., 0., 0., 0. ] if i < 50 else [0.1, 0., 0., 0., 0., 0., 0.0])
        images.append(env.render('rgb_array', spp=1))


    animate(images)