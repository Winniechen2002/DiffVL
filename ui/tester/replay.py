import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import argparse
import tqdm
from llm.envs import MultiToolEnv
from llm.tiny import Scene
from llm.ui import GUI
from tools.utils import animate


def main():
    gui = GUI()

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()
    if args.path.endswith(".th"):
        args.path = args.path[:-3]

    env = MultiToolEnv()

    state = torch.load(args.path + ".th")

    env.set_state(state)
    action = torch.load(args.path + ".action")

    if args.output is not None:
        images = []
        images.append(env.render('rgb_array'))
        for i in tqdm.tqdm(action, total=len(action)):
            env.step(i)
            images.append(env.render('rgb_array', spp=1))

        animate(images, filename=args.output, fps=30)
    else:
        scene = Scene(env)

        gui.load_scene(scene)
        gui._remaining_actions = list(action)

        gui.start()


if __name__ == '__main__':
    main()