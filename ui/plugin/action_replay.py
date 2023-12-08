import torch
import numpy as np
from sapien.core import Pose
from .plugin_base import Plugin

from transforms3d import quaternions 
from pytorch3d.transforms import quaternion_apply
from tools.utils import totensor
from .control_utils import control3d


class ActionReplayer(Plugin):
    mode_name = 'replay_action'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = None
        self.N = None
        self.i = None
        self.tim = 0

    def rendering(self):
        if self.viewer.window.key_press('n'):# and self.selected:
            self._trigger()

    def enter_mode(self):
        self.action = torch.load('sketch/windrope/best_action')['actions']
        self.N = self.action.shape[0]
        self.i = -1
        self.tim = -1

    def get_object(self):
        #for i in self.engine._soft_bodies.values():
        #    return i
        return self.viewer.selected_softbody

    def leave_mode(self):
        self.action = None
        self.N = None
        self.i = None
        self.tim = 1

    def step(self, action):
        # if self.update is not None:
        #     # raise NotImplementedError
        #     state = self.engine._env.get_state()
        #     for i, k in self.engine._soft_bodies.items():
        #         state.X[k['indices'].detach().cpu().numpy()] = self.changes[i][:, [0, 2, 1]]
        #     self.engine._env.set_state(state)

        #     self.next_shape = None
        #     self.update = None
        # return action, None
        if self.tim % 5 != 0 or self.action is None or self.i is None:
            return action, None
        else:
            return self.action[self.i], None

    def monitor(self):
        from .control_utils import normal_monitor
        normal_monitor(self.viewer)

        self.tim += 1
        if self.tim % 5 == 0:
            self.i += 1
            if self.i >= self.N:
                self.leave_mode()
                self.viewer.enter_mode("normal")
            self.tim = 0
