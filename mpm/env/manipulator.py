import gym
import numpy as np

from .env_module import EnvModule
from .geom import compute_bbox
from tools.config import Configurable, as_builder, CN


@as_builder
class Manipulator(Configurable,
                  EnvModule):
    def __init__(self, cfg=None):
        super(Manipulator, self).__init__()
        self.cfg = cfg

    def empty_state(self):
        return [np.array([0., 0., 0., 1., 0., 0., 0.])
                for i in range(self.env.simulator.n_bodies)]

    def move_to(self):
        raise NotImplementedError

    def sample_placement(self):
        raise NotImplementedError

    def postprocessing_actions(self, actions):
        return actions

    @property
    def action_space(self):
        raise NotImplementedError(f"{self.__class__}'s action space is not implemented.")

    def get_initial_action(self):
        action = np.random.normal(size=(12,)) * 0.001 # for rotation, this value must be greater than zero to gain gradient..
        return action
        #return np.zeros(12)

    def step(self, action, *args, **kwargs):
        self.env._step(action, *args, **kwargs)


class ParallelGripper(Manipulator):
    def __init__(
        self,
        cfg=None,
        size=(0.03, 0.15, 0.15),
        friction=5.,
        lv=(0.01, 0.01, 0.01),
        av=(0.01, 0.01, 0.01),
        h=0.06,
        r=0.02,
        default_type=1,
        stiffness=0.,
        n_bodies=2,
        round=0.,
    ):
        super(ParallelGripper, self).__init__()

    def _update_cfg(self, cfg: CN):
        self.sim_cfg = cfg
        cfg.PRIMITIVES = cfg.PRIMITIVES[:self._cfg.n_bodies]
        for c in cfg.PRIMITIVES[:2]:
            c['friction'] = self.cfg.friction
            c['action']['scale'] = self.cfg.lv + self.cfg.av
            c['stiffness'] = self.cfg.stiffness

            if c['shape'] == 'Box':
                c['size'] = self.cfg.size
                c['round'] = self.cfg.round
            elif c['shape'] == 'Compositional':
                c['shapes']['Box']['size'] = self.cfg.size
                c['shapes']['Capsule']['r'] = self.cfg.r
                c['shapes']['Capsule']['h'] = self.cfg.h
                c['default_type'] = 1
        return cfg

    def switch_type(self, t):
        for cfg, i in zip(self.sim_cfg.PRIMITIVES, self.env.primitives):
            if cfg['shape'] == 'Compositional':
                i.type[None] = t

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, (12,))

    def sample_grasp_pos(self, x, gap=0.03):
        bbox = compute_bbox(x)
        center = (bbox[0] + bbox[1]) / 2

        s = self.cfg.size[0]

        state = self.empty_state()

        state[0][:3] = np.array(
            [
                bbox[0][0] - gap - s,
                center[1],
                center[2]
            ]
        )

        if len(state) > 1:
            state[1][:3] = np.array(
                [
                    bbox[1][0] + gap + s,
                    center[1],
                    center[2]
                ]
            )

        return list(state)
