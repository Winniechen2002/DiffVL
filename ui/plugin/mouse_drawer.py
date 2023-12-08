import numpy as np
import copy
import torch
import random
from sapien.core import Pose
from envs.soft_utils import rgb2int

from envs.world_state import WorldState
from .plugin_base import Plugin

from transforms3d import quaternions 
from pytorch3d.transforms import quaternion_apply
from tools.utils import totensor
from .control_utils import control3d

from llm.tiny.softbody import SoftBody
from sapien.core import renderer as R


class MouseDrawer(Plugin):
    # currently there is only one shape
    mode_name = 'mouse_draw'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        self.next_shape = None
        self.up = 0.
        self.id = None
        self.las = None
        self.cnt_objs = 0
        
    def rendering(self):
        if self.viewer.window.key_press('z'):# and self.selected:
            print('drawing with pen')
            self._trigger()

    def enter_mode(self):   
        self._old_paused = self.viewer.paused
        self._next_shape = None
        self.viewer.toggle_pause(True)
        self.id = None

    def leave_mode(self):
        if self.viewer.paused:
            # recover ..
            self.viewer.toggle_pause(self._old_paused)

    def step(self, action):
        return action, None

    def get_color(self):
        self.cnt_objs += 1
        if self.cnt_objs % 3 == 0:
            return rgb2int(255, 0, 0)
        elif self.cnt_objs % 3 == 1:
            return rgb2int(0, 255, 0)
        elif self.cnt_objs % 3 == 2:
            return rgb2int(0, 0, 255)

    def monitor(self):
        from .control_utils import normal_monitor, track_mouse
        from envs.soft_utils import sphere
        # normal_monitor(self.viewer)

        self.target = track_mouse(self, lambda: self.up) #float(self.engine.obs['qpos'][1]))
        wx, wy = self.viewer.window.mouse_wheel_delta
        # print(wx)
        if wx != 0:
            self.up += wx * 0.02
        # print(self.target)

        if self.target is not None:
            target = self.target[0]
            if self.las is None or target[0] != self.las[0] or target[1] != self.las[1]:

                self.las = target
                pos = (target[0], self.up + 0.01, target[1])

                stt = self.engine._env.get_state()
                if self.id is None:
                    self.id = np.unique(stt.ids)[0] + 1
                out = sphere(0.01, center=pos, n = None)
                new = WorldState.get_empty_state(n=len(out))
                new.X[:] = out
                new.ids[:] = self.id
                new.color[:] = self.get_color()
                self.update_scene_by_state(stt.add_state(new))
                    
        if self.viewer.window.mouse_click(0):
            print('click')
            self.viewer.select(False)
            self.leave_mode()
            self.viewer.enter_mode("normal")
