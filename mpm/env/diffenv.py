import copy
import os
import torch
from torch import nn
import numpy as np
from tools.config import Configurable, merge_inputs, parse_args

from .manipulator import Manipulator, ParallelGripper
from .observations import Observer, TaichiRGBDObserver
from .solver_utils import Solver, np2th
from .geom import rgb2int
from tools import CN

FILEPATH = os.path.dirname(os.path.abspath(__file__))


def get_default_cfg(cfg_path, sim_cfg=None):
    if cfg_path[0] != '/' and cfg_path.startswith('configs'):
        cfg_path = os.path.join(FILEPATH, cfg_path)
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    if sim_cfg is not None:
        cfg.defrost()
        cfg.SIMULATOR = merge_inputs(cfg.SIMULATOR, **sim_cfg)
        cfg.freeze()
    return cfg



class DifferentiablePhysicsEnv(Configurable):
    def __init__(self,
                 cfg=None,
                 cfg_path='configs/plb_cuda.yml',
                 observer_cfg=Observer.to_build(TYPE=TaichiRGBDObserver),
                 manipulator_cfg=Manipulator.to_build(TYPE=ParallelGripper),
                 warning_off=True,
                 device='cuda:0',
                 sim_cfg=None,
                 reward_type='dense',
                 ):
        assert reward_type == 'dense'

        if warning_off:
            import warnings
            warnings.filterwarnings('ignore')

        super(DifferentiablePhysicsEnv, self).__init__()

        self.observer: Observer = Observer.build(
            cfg=observer_cfg
        )

        self.manipulator: Manipulator = Manipulator.build(
            cfg=manipulator_cfg
        )

        simulator_cfg = get_default_cfg(cfg_path, sim_cfg=sim_cfg)
        simulator_cfg = self.manipulator.update_cfg(simulator_cfg)
        simulator_cfg = self.observer.update_cfg(simulator_cfg)
        simulator_cfg = self._update_cfg(simulator_cfg)

        # print(simulator_cfg.SIMULATOR)

        from mpm.cuda_env import CudaEnv
        from mpm.torch_wrapper import DiffModel
        taichi_env = CudaEnv(simulator_cfg)
        self.torch_env = DiffModel(taichi_env)

        self.taichi_env = taichi_env
        self.simulator = self.taichi_env.simulator
        self.renderer = self.taichi_env.renderer
        # self.primitives = self.taichi_env.primitives
        self.substeps = self.simulator.substeps

        self.manipulator.set_env(self)
        self.observer.set_env(self)

        self.solver = None

        self._requires_grad = False
        self.device = self._default_device = device

        self._obs = None
        self._idx = 0
        self.rl_step = 0

    @property
    def action_space(self):
        return self.manipulator.action_space

    @property
    def observation_space(self):
        return self.observer.observation_space

    def _update_cfg(self, cfg):
        return cfg


    def get_tool_state(self, index=0):
        return self.simulator.get_tool_state(index)

    def set_tool_state(self, index, tools):
        self.simulator.set_tool_state(index, tools)
        self._obs = None

    def get_state(self, index=0):
        tmp = self.simulator.get_state(index)
        state = {
            'particles': tmp[:4],
            'tools': tmp[4:],
            'softness': self.simulator.get_softness(),
            'n': len(tmp[0]),
            'ids': self.simulator.get_object_id(device='numpy'),
        }
        state['color'] = self.taichi_env.particle_colors.copy()
        state['requires_grad'] = self._requires_grad
        return state

    @staticmethod
    def add_state(A, B):
        import copy
        out = copy.copy(A)
        out['particles'] = [np.concatenate((a, b)) for a, b in zip(A['particles'], B['particles'])]
        out['n'] = A['n'] + B['n']
        out['color'] = np.concatenate((A['color'], B['color']))
        out['ids'] = np.concatenate((A['ids'], B['ids']))
        return out

    def set_state(self, state, index=0, device=None, **kwargs):
        if 'requires_grad' in state:
            self._requires_grad = state['requires_grad']
        if 'requires_grad' in kwargs:
            self._requires_grad = kwargs['requires_grad']

        assert state['n'] == len(state['particles'][0])
        assert index == 0, "One can only modify the initial state .."
        self.simulator.set_state(index, list(state['particles']) + list(state['tools']))
        # self.simulator.set_object_id(state['ids'])
        self.simulator.set_softness(state['softness'])
        self.taichi_env.particle_colors = state['color']
        self.device = device or self._default_device
        self.clear(**kwargs)

    def clear(self, **kwargs):
        self._idx = 0
        self._obs = None
        if self._requires_grad:
            self.torch_env.device = self.device
            self.torch_env.zero_grad(**kwargs)
            self._torch_obs = self.torch_env.get_obs(0, self.device)
            self._obs = self.wrap_obs(self._torch_obs)

    @staticmethod
    def _empty_soft_body_state(
            n=None,
            init=None,
            color=rgb2int(127, 127, 127)
    ):
        assert n is not None or init is not None

        if init is None:
            init = np.zeros((n, 3))
        else:
            n = len(init)

        init_v = np.zeros((n, 3))
        F = np.eye(3)[None, :].repeat(n, axis=0)
        C = np.zeros((n, 3, 3))

        particles = [init, init_v, F, C]
        return {
            'particles': particles,
            'is_copy': True,
            'softness': 666.,
            'color': np.zeros(n, dtype=np.int32) + color,
            'n': n,
            'ids': np.zeros(n, dtype=np.int32)
        }

    def empty_state(self, *args, **kwargs):
        state = self._empty_soft_body_state(*args, **kwargs)
        state['tools'] = self.manipulator.empty_state()
        return state
 
    @property
    def cur_step(self):
        return self._idx * self.substeps

    def step(self, *args, **kwargs):
        self.manipulator.step(*args, **kwargs)
        obs = self.observer.get_obs()

        taichi_state = self._get_obs()
        self.rl_step += 1
        if hasattr(self, "_loss_fn"):
            loss = self._loss_fn(self.rl_step - 1, **taichi_state)
            if isinstance(loss, tuple):
                loss, info = loss
            else:
                info = {}
            info['loss'] = loss
            done = self.rl_step >= self._cfg.max_episode_steps
            if done:
                info['TimeLimit.truncated'] = True
        else:
            loss, done, info = -99., None, None

        return obs, -float(loss), done, info

    def render(self, mode='plt', **kwargs):
        return self.observer.render(mode, **kwargs)

    def reset(self, requires_grad=False, initial_state_sampler=None, loss_fn=None):
        if initial_state_sampler is not None:
            self._loss_fn = loss_fn
            init_state = initial_state_sampler(self)
            init_state['requires_grad'] = requires_grad
            self.set_state(init_state)
        elif self.task is not None:
            init_state = self.task.reset(self, requires_grad=requires_grad)
        else:
            print("task not specified")

        self.rl_step = 0
        return self.observer.get_obs()


    # ----------------- code for backward -----------------------------
    def zero_grad(self):
        assert self._requires_grad
        if self.cur_step > 0:
            self.set_state(self.get_state(self.cur_step))

    def wrap_obs(self, obs):
        return self.torch_env.wrap_obs(obs)

    def _step(self, action, pos_rot=None):
        if not self._requires_grad:
            if action is not None:
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                else:
                    action = np.array(action)
            self.simulator.step(action=action, pos_rot=pos_rot)
            self._obs = None
        else:
            action = action[:] # maybe parameter to torch tensor
            assert isinstance(action, torch.Tensor)
            nxt_step = (self._idx + 1) * self.substeps
            assert nxt_step < self.simulator.max_steps,\
                f"{nxt_step} exceed max steps {self.simulator.max_steps} in grad mode"
            self._torch_obs = self.torch_env.forward(self._idx, action, *self._torch_obs, pos_rot=pos_rot)
            self._obs = self.wrap_obs(self._torch_obs)
            self._idx += 1

    def _render(self, **kwargs):
        index = self.cur_step
        if self._cfg.use_taichi:
            x = self.simulator.get_x(index)
            # self.renderer.set_particles(x, self.taichi_env.particle_colors)
            img = self.renderer.render_frame(frame_id=index, **kwargs)
        else:
            # self.simulator.set_color(self.taichi_env.particle_colors)
            img = self.renderer.render(self.simulator, self.simulator.states[index], **kwargs)
        if img.shape[-1] >= 3:
            img[:, :, :3] = np.uint8(img[:, :, :3].clip(0, 1) * 255)
        return img

    def _get_obs(self):
        if self._obs is None:
            self._obs = {
                'pos': self.simulator.get_x(self._idx, device=self.device),
                'vel': self.simulator.get_v(self._idx, device=self.device),
                'dist': self.simulator.get_dists(self._idx, device=self.device),
                'tool': torch.stack(self.simulator.get_tool_state(self._idx, device=self.device))
            }
        return self._obs

    def execute(self, initial_state, actions, filename=None, render_freq=1, **kwargs):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        initial_state = copy.copy(initial_state)
        initial_state['requires_grad'] = False
        self.set_state(initial_state)

        if filename is not None:
            images = [self.render('array')]

        ran = enumerate(actions)

        for idx, act in ran:
            self.step(act)
            if filename is not None and idx % render_freq == 0:
                img = self.render('array')
                images.append(img)

        if filename is not None:
            from ..utils import animate
            if filename.endswith('.mp4'):
                return animate(images, filename, **kwargs)
            else:
                return images
        else:
            return self.get_state()

    def solve(self, initial_state, initial_action, loss_fn=None, **kwargs):
        loss_fn = loss_fn or self._loss_fn
        if self.solver is None:
            self.solver = Solver(**kwargs)
        return self.solver.solve(self, initial_state, initial_action, loss_fn, **kwargs)

    def mpc(self, initial_state, initial_action, **kwargs):
        from .mpc_utils import MPC
        mpc = MPC()
        return mpc.mpc(self, state=initial_state, initial_actions=initial_action, **kwargs)