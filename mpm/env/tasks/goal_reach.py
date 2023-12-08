import torch
import matplotlib.pyplot as plt
import copy
from .task import Task
from .. import geom
import argparse
import numpy as np

class GoalReach(Task):
    def __init__(
        self, state, goal, cfg=None, shape_weight=1., contact_weight=1.,
    ) -> None:
        super().__init__()
        self.init(state, goal)

    def init(self, state, goal):
        self.state = copy.copy(state)
        self.goal = goal
        self.initial_action = None
        # let's only single object first
        self.state['ids'] = np.ones_like(self.state['ids']) 


    def reset(self, env, requires_grad=True):
        # copied from task.py
        env.task = self
        self.env = env
        self.env._loss_fn = self._loss_fn
        self.device = self.env.device

        ids = self.state['ids']
        self.ctrl_mask = geom.vec(ids>0, device=self.device, dtype=torch.bool)
        self.bg_mask = None

        self.state['requires_grad'] = requires_grad
        self.env.set_state(
            self.state,
            return_grid=(1,),
            return_svd=True,
        )

        self._taichi_state = self.env._get_obs()
        self._init_objs = self.shape_from_state(self._taichi_state)

        # self.set_goal()
        self.goals = self.shape_from_state({"pos": geom.vec(self.goal[np.random.choice(len(self.goal), int(self.ctrl_mask.sum()))], device=self.device)})

        return copy.copy(self.state)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_goal")
    parser.add_argument("--env_type", default=None)
    parser.add_argument("--h", default=50, type=int)
    parser.add_argument("--solver", default='solver', choices=['solver', 'mpc'])
    args, _ = parser.parse_known_args()

    from ..env_hubs import comp_env
    assert args.env_type is None
    import pickle
    with open(args.state_goal, 'rb') as f:
        state, goal = pickle.load(f)

    env = comp_env()
    task = GoalReach(state, goal)
    task.solve_it(env, T=args.h, parser=parser)


if __name__ == '__main__':
    main()