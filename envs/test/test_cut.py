from envs.plb import MultiToolEnv
from tools.utils import animate
import tqdm

quality = 2
from envs import soft_utils
soft_utils.N_VOLUME = 40000
env = MultiToolEnv(sim_cfg=dict(ground_friction=1., yield_stress=200., dx=1./64/quality, dt=0.00005/quality, substeps=20 * 2 * quality, vol=(1./64/quality/2)**3, mass=(1./64/quality/2)**3, n_particles=40000))
print(env.simulator.dt)

import numpy as np

from envs.world_state import WorldState
from envs.test_utils import init_scene

init_scene(env, 0)
state: WorldState = env.get_state()
state.E_nu_yield[:, 0] = 100
state.E_nu_yield[:, 2] = 30

state = state.switch_tools('Gripper', np.array([0.5, 0.3, 0.5, 0., 0., 0., 0.1]), friction=0., size=(0.02, 0.2, 0.2), action_scale=(0.015, 0.015, 0.015, 0., 0., 0.0, 0.015), softness=0.)
images = []
env.set_state(state)

for i in tqdm.trange(100):
    env.step([0, -0.4, 0., 0., 0., 0., 1. if i > 25 else 0.])
    images.append(env.render('rgb_array', spp=1))


animate(images)