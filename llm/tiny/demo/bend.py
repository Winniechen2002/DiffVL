import torch
import argparse
import numpy as np 
from llm.envs import MultiToolEnv, WorldState
from tools.utils import animate
#from llm.envs.soft_utils

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = MultiToolEnv(sim_cfg=dict(yield_stress=1000., gravity=(0., -0.2, 0.), ground_friction=0.))

    state = WorldState.sample_shapes(method='box', center=(0.5, 0.14, 0.5), shape_args=(0.3, 0.06, 0.06), color=(255, 0, 0)).switch_tools("Gripper", [0.5, 0.05, 0.5, 0., 0.2])

    env.set_state(state)
    for i in range(20):
        env.step([0., 0., 0., 0., 0.])
    
    images = []
    for i in range(35):
        env.step([0., -0.3, 0., 0., -1.])
        images.append(env.render('rgb_array'))

    animate(images, 'bend.mp4')


if __name__ == '__main__':
    main()