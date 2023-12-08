import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from llm.ui import GUI
from llm.tiny import Scene
from llm.envs import MultiToolEnv, WorldState
from llm.envs.test_utils import init_scene


def main():
    gui = GUI()

    env = MultiToolEnv(gripper_cfg=dict(size=(0.03, 0.1, 0.1)))

    state = WorldState.sample_shapes(
        childs=[
            dict(method='box', center=(0.5, 0.14, 0.3), shape_args=(0.3, 0.1, 0.1), color=(0, 0, 255)),
        ]).switch_tools("Gripper", [0.5, 0.05, 0.5, 0., 0.2])


    env.set_state(state)
    for i in range(50):
        env.step(env.action_space.sample() * 0)

    gui.load_scene(Scene(env))
    gui.start()




if __name__ == "__main__":
    main()