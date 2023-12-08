from llm.envs import MultiToolEnv, WorldState
from llm.envs import test_utils as tu
import numpy as np


env = MultiToolEnv(sim_cfg=dict(max_steps=1024))
tu.init_scene(env, 0)
state = env.get_state()
state = state.switch_tools('Gripper', np.array([0.5, 0.1, 0.5, 0., 0.2]))

import torch
from torch import nn, optim
from tools.utils import totensor

# torch.autograd.set_detect_anomaly(True)

actions = nn.Parameter(totensor(np.zeros((50, 5)), 'cuda:0'), requires_grad=True)
adam = optim.Adam([actions], lr=0.01)

import tqdm
for i in tqdm.trange(100):
    adam.zero_grad()
    env.set_state(state, requires_grad=True)
    loss = 0
    for i in range(50):
        env.step(actions[i])
        obs = env.get_obs()
        loss = loss - obs['pos'].mean(axis=0)[1] + torch.relu(obs['dist'].min(axis=0).values).sum()

    print(loss)
    loss.backward()
    adam.step()