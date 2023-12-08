# currently can only be run on GPU1
import torch
from tools.utils import animate
from gym_env import GymEnv

env = GymEnv(0)

import os

img = env.render()
print(env.action_space)


images = []
for i in range(100):
    env.step([1] + [0] * 6)
    img = env.render()
    images.append(img)

animate(images)