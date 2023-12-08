import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from llm.tiny import Scene
from envs import MultiToolEnv
from envs.test_utils import init_scene
from ui import GUI
from ui.async_env import AsyncPlbEnv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=int, default=0)
    parser.add_argument("--state", type=str, default=None)
    args = parser.parse_args()

    gui = GUI()

    env = MultiToolEnv(sim_cfg=dict(max_steps=100))

    if args.state is not None:
        import torch
        state = torch.load(args.state)
        env.set_state(state)
    else:
        init_scene(env, args.scene)
    scene = Scene(env)
    #env.render('plt')

    gui.load_scene(scene)
    gui.start()





if __name__ == '__main__':
    main()
