import torch
import cv2
import argparse
import numpy as np 
from llm.envs import MultiToolEnv, WorldState
from tools.utils import animate
from llm.envs.soft_utils import rgb2int

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = MultiToolEnv(sim_cfg=dict(yield_stress=1000., gravity=(0., -1., 0.), ground_friction=3.))

    state = WorldState.sample_shapes(method='sphere', center=(0.5, 0.4, 0.5), shape_args=(0.05,), color=(255, 0, 0)).switch_tools("Gripper", [0.5, 0.4, 0.5, np.pi/4, 0.035])

    env.set_state(state)
    cv2.imwrite('out.png', env.render('rgb_array')[..., ::-1])

    for i in range(20):
        env.step([0., 0., 0., 0., 0.])
    
    images = []
    for i in range(100):
        obs = env.step([1. if i < 10 else 0., 0., 0., 0., 0.01 if i < 5 else (1. if i < 10 else 0.)])[0]
        images.append(env.render('rgb_array', spp=1))
    #env.render('plt')

    animate(images, 'throw.mp4')


if __name__ == '__main__':
    main()