import os
import tqdm
import cv2
from llm.envs import MultiToolEnv
from llm.optim.loader import load_scene
from llm.genetic.utils import GENOMEPATH
from tools.utils import animate

env = MultiToolEnv(sim_cfg=dict(max_steps=100, ground_friction=0.8))

state = load_scene(os.path.join(GENOMEPATH, '../../libraries/config/box_pick.yml'))
env.set_state(state)

images = []

for i in tqdm.trange(300):
    images.append(env.render('rgb_array', spp=1))
    if i < 100:
        env.step([0., 0., 0., 1., 0., 0., 1.])
    elif i < 200:
        env.step([0., 0., 0., 0., 1., 0., 1.])
    else:
        env.step([0., 0., 0., 0., 0., 1., 1.])
animate(images)