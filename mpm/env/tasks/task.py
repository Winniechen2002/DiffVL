"""[summary]
Let's slightly simplify the concept
The loss always have the following forms: 
- modifying one part of the objects 
- and fixing the remaning parts

Thus, even if the shape may have multiple objects, we only consider single object.
It would be exciting if we are able to do global search with this, but it seems infeasible.
"""
import torch
import gym
import numpy as np
from .. import geom
from ..geom import randp, random, default_shape_metrics, SoftObject
from tools import Configurable, as_builder, CN, merge_inputs
import argparse


@as_builder
class Task(Configurable, gym.Env):
    def __init__(
        self,
        cfg=None,

        #foreground object ids
        # it could be a None, tuple, or single integer
        ctrl_object_id=None,

        # parameters to compare shapes
        shape_metric=default_shape_metrics(),

        # parameters for shape generator
        center=randp((0.5, 0.1, 0.5)),
        n=None,
        method='box',
        childs=None,
        shape_args=None,  # shape args should be tuple to define the random genereator
        object_id=None,

        bg_weight=1.,

    ) -> None:
        super().__init__()

        from ..diffenv import DifferentiablePhysicsEnv as DiffEnv
        self.env: DiffEnv = None

    @property
    def ctrl_object_id(self):
        return self._cfg.ctrl_object_id 

    def update_cfg(self, cfg):
        self._cfg.merge_from_other_cfg(cfg)

    def shape_metrics(self, obj, goal, **kwargs):
        # compare the shape between the two ..
        shape_metric = merge_inputs(self._cfg.shape_metric, **kwargs)
        return geom.distance_to(obj, goal, shape_metric)

    def set_goal(self):
        raise NotImplementedError

    def reset(self, env, requires_grad=True):
        from ..diffenv import DifferentiablePhysicsEnv as DiffEnv
        env.task = self
        self.env: DiffEnv = env
        self.env._loss_fn = self._loss_fn
        self.device = self.env.device

        # sample shapes
        p, ids, _, color = geom.sample_objects(self._cfg)

        # store the id for foreground and background
        ctrl_object_id = self.ctrl_object_id
        if ctrl_object_id is None:
            ids = np.ones(len(ids), dtype=np.int32)
        elif isinstance(self.ctrl_object_id, int):
            ids = np.int32(ids == ctrl_object_id)

        else:
            raise NotImplementedError
        self.ctrl_mask = geom.vec(ids>0, device=self.device, dtype=torch.bool)
        # print(self.ctrl_mask)
        self.bg_mask = torch.logical_not(self.ctrl_mask)
        if not self.bg_mask.any():
            self.bg_mask = None

        # finish the simulation state
        state = self.env.empty_state(init=p)
        #TODO: we may need adjust the way to sample manipulators..
        state['tools'] = self.env.manipulator.sample_grasp_pos(p[ids>0]) 
        state['ids'] = ids
        state['color'] = color
        self._initial_state = state

        # set the environment state 
        state['requires_grad'] = requires_grad
        return_grids = (1,)
        if self.bg_mask is not None:
            return_grids += (0,)

        self.env.set_state(
            state,
            return_grid=return_grids,
            return_svd=True,
        )

        self._taichi_state = self.env._get_obs()
        self._init_objs = self.shape_from_state(self._taichi_state)
        self.set_goal()

        return state


    def _loss_fn(self, cur_step, **kwargs):
        self._cur_step = cur_step
        self._taichi_state = kwargs

        losses = self.compute_loss()

        loss = 0
        info = {}
        for k, v in losses.items():
            weight = self._cfg[k+'_weight']
            if isinstance(v, tuple):
                v, other = v
                info.update(other)

            if weight > 0.:
                loss = loss + v * weight
                info[k] = float(v)

        return loss, info

    @property
    def cur_objects(self):
        return self.shape_from_state(self._taichi_state)

    def shape_from_state(
        self,
        state, # observation from the taichi environment 
    ):
        # we directly create the foreground and background objects from the state
        objs = {}
        names = ['bg', 'ctrl']
        for idx, mask in enumerate([self.bg_mask, self.ctrl_mask]):
            if mask is not None:
                #assert isinstance(state['grid'], dict)
                grid = state['grid'][idx] if 'grid' in state else None
                kk = {i: state[i][mask] for i in state if i != 'tool' and i != 'grid'}
                objs[names[idx]] = SoftObject(
                    env=self.env,
                    grid=grid,
                    **kk
                )
        return objs 

    def set_goal():
        raise NotImplementedError

    def render_goal(self, *args, **kwargs):
        from ..diffenv import rgb2int
        previous_state = self.env.get_state()

        inp = [geom.to_numpy(self.goals[k].pos) for k in self.goals]
        colors = [rgb2int(255, 127, 127), rgb2int(127, 127, 255)]
        colors = [np.zeros(len(i)) + c for i, c in zip(inp, colors)]
        new_state = self.env._empty_soft_body_state(
            init=np.concatenate(inp), color=np.concatenate(colors))

        new_state['tools'] = previous_state['tools']
        self.env.set_state(new_state)
        # print('start rendering goal..')
        image = self.env.render(*args, **kwargs, primitive=0)

        self.env.set_state(previous_state)
        return image


    def compute_loss(self):
        objs = self.cur_objects

        output = {
            'shape': self.shape_metrics(
                objs['ctrl'], self.goals['ctrl'],
            ),
            'contact': objs['ctrl'].dist_to_tool().sum(),
        }

        if 'bg' in self.goals:
            output['bg'] = self.shape_metrics(
                objs['bg'], self.goals['bg'], center=0., # we do not allow centers..
            )

            raise NotImplementedError("Not Implemented Yet")

        return output


    def solve_it(self, env, init_actions=None, T=None, parser: argparse.ArgumentParser=None):
        # utils..
        import os.path as osp
        import matplotlib.pyplot as plt
        from ..solver_utils import Solver
        from ..mpc_utils import MPC

        if init_actions is None:
            if T is None: T= 50
            init_actions = [env.manipulator.get_initial_action() for i in range(T)]


        args, _ = parser.parse_known_args()
        if args.solver == 'solver':
            solver = Solver.parse(
                render_path='task',
                render_interval=50,
                max_iter=1000,
                lr=0.01,
                parser=parser,
            )
            env.solver = solver
        else:
            solver = MPC.parse(
                render_path='task',
                num_iters=20,
                horizon=20,
                lr=0.02,
                parser=parser
            ) 
            print(solver)

        state = self.reset(env)

        render_path = solver._cfg.render_path
        if render_path is not None:
            import os
            os.makedirs(render_path, exist_ok=True)
            plt.imshow(env.render('rgb_array'))
            plt.savefig(osp.join(render_path, 'init.png'))

            plt.imshow(self.render_goal('rgb_array'))
            plt.savefig(osp.join(render_path, 'goal.png'))
        env.task = self

        if args.solver == 'solver':
            env.solve(state, init_actions)
        else:
            solver.mpc(env, state, init_actions)