import os
import tqdm
import numpy as np
from tools.config import CN
from llm.genetic.init_state import load_scene
from llm import LLMPATH
from llm.envs import MultiToolEnv, MPMSimulator
from llm.envs.plb import Renderer

env = MultiToolEnv(sim_cfg=dict(max_steps=50))

cfg_path = os.path.join(LLMPATH, 'genetic', 'dataset', 'config/box.yml')

state =  load_scene(cfg_path)
env.set_state(state)

env2 = MultiToolEnv(sim_cfg=dict(max_steps=40))
env2.set_state(state)


images = []
actions = np.random.random((50, *env.action_space.shape)) * 2 - 1

for i in tqdm.tqdm(actions, total=len(actions)):
    i[1] = -0.3
    images.append(np.concatenate([env.render(mode='rgb_array'), env2.render(mode='rgb_array')], axis=1))
    #images.append(env.render(mode='rgb_array'))
    env.step(i)
    env2.step(i)
from tools.utils import animate
animate(images, fps=10)