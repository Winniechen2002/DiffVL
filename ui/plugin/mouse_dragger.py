import numpy as np
import copy
import math
import torch
import random
from sapien.core import Pose
from envs.soft_utils import rgb2int

from envs.world_state import WorldState
from .plugin_base import Plugin

from transforms3d import quaternions 
from pytorch3d.transforms import quaternion_apply
from tools.utils import totensor
from .control_utils import control3d, track_mouse_mover

from llm.tiny.softbody import SoftBody
from sapien.core import renderer as R


class MouseDragger(Plugin):
    # currently there is only one shape
    mode_name = 'mouse_drag'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        self.next_shape = None
        self.velocity = None
        self.las_pos = None
        self.id = None
        self.N = None
        self.update = False
        self.sensitivity = 2
        self.range = 0.01
        self.eps = 1e-5
        self.color_copy = None

    def collect_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity

    def collect_range(self, range):
        self.range = range

    def build_windows(self):
        self.window = (
            R.UIWindow()
            .Pos(410, 10)
            .Size(200, 400)
            .Label("Mouse Drag")
            .append(
                R.UIDisplayText().Text("Sensitivity of dragger:"),
                R.UISliderFloat()
                .Min(0.1)
                .Max(20)
                .Value(self.sensitivity)
                .Label("Sensitivity")
                .Callback(lambda wc: self.collect_sensitivity(wc.value)),
                R.UIDisplayText().Text("Range of dragger:"),
                R.UISliderFloat()
                .Min(0.0001)
                .Max(0.03)
                .Value(self.range)
                .Label("Range")
                .Callback(lambda wc: self.collect_range(wc.value)),
            )
        )
        return self.window

    def rendering(self):
        if self.viewer.window.key_press('c'):# and self.selected:
            self.id, self.N = self.get_itsc()
            self.update = True
            print("render",self.id,len(self.id))
            if len(self.id) > 0:
                print('dragging with mouse')
                stt = self.engine._env.get_state()
                self.color_copy = stt.color[self.id]
                # print("copy",self.color_copy)
                stt.color[self.id] = rgb2int(255,255,255)
                self.update_scene_by_state(stt)
                self._trigger()

    def leave_mode(self):
        if self.N and len(self.id) > 0:
            stt = self.engine._env.get_state()
            stt.color[self.id] = self.color_copy
            self.update_scene_by_state(stt)
        if self.viewer.paused:
            # recover ..
            self.viewer.toggle_pause(self._old_paused)

    def get_itsc(self):
        # get_itsc continue computing
        from .control_utils import track_mouse_mover
        #print("qpos",self.engine.obs['qpos'][1])
        self.viewer.select()
        # print(self.get_object())
        self.las_pos, ray, pos = track_mouse_mover(self)
        stt = self.engine._env.get_state()
        print(stt.color)
        N = stt.X.shape[0]
        
        pos[[1, 2]] = pos[[2, 1]]
        ray[[1, 2]] = ray[[2, 1]]
        print(pos, self.las_pos)
        #raise NotImplementedError("bugs here")
        
        tx = stt.X - pos
        
        dir = ray / math.sqrt((ray ** 2).sum().item())
        
        prj = (tx * dir).sum(axis=1)
        # first_point = stt.X[np.argmin(prj)]
        # print("first point",first_point)
        dis = (tx * tx).sum(axis=1) - prj ** 2
        # dis = ((stt.X-first_point) * (stt.X-first_point)).sum(axis=1)
        tmp = np.where(dis < self.eps)
        if len(tmp[0]) == 0:
            self.leave_mode()
            return tmp[0], 0
        #print("tmp: ",tmp,len(tmp[0]))
        minn = tmp[0][np.argmin(prj[tmp[0]])]
        # print("minn ",minn, stt.X[minn])
        first_point = stt.X[minn]
        dis = ((stt.X-first_point) * (stt.X-first_point)).sum(axis=1)
        tmp = np.where(dis < self.range)
        #print("tmp: ",tmp,len(tmp[0]))
        return tmp[0], N


    def enter_mode(self):   
        self._old_paused = self.viewer.paused
        self._next_shape = None
        self.velocity = None
        assert len(self.id) > 0

    def get_object(self):
        #for i in self.engine._soft_bodies.values():
        #    return i
        return self.viewer.selected_softbody

    def step(self, action):
        if self.update:
            # difficult to use
            # stt = self.engine._env.get_state()
            # stt.color[self.id] = rgb2int(255,255,255)
            # self.update_scene_by_state(stt)
            self.update = False
        return action, self.velocity

    def monitor(self):
        from .control_utils import track_mouse_pos
        from envs.soft_utils import sphere
        # print("qpos",self.engine.obs['qpos'][1])
        self.target = track_mouse_pos(self)
        wx, wy = self.viewer.window.mouse_wheel_delta
        
        if self.target is None:
            self.velocity = None
        else:
            dir = np.array([self.target[0] - self.las_pos[0], self.target[1] - self.las_pos[1]])
            srt = math.sqrt(dir[0] * dir[0] + dir[1] * dir[1])
            if srt != 0: 
                dir = dir / srt * 0.1
            # if dir[0] != 0 or dir[1] != 0 or wx != 0:
            #     print(dir[0], dir[1], wx * 0.02)

            self.las_pos = self.engine.last_scene.initial_state.X[self.id[0]]
            self.las_pos = [self.las_pos[0], self.las_pos[2]]
            # self.las_pos = self.target
            self.velocity = np.zeros((self.N, 3))
            self.velocity[self.id] += np.array([dir[0], wx/10, dir[1]]) * self.sensitivity
            #print(self.velocity[self.id])

        #print(self.velocity)

        if self.viewer.window.mouse_click(0):
            print('click')
            self.viewer.select(False)
            self.leave_mode()
            self.viewer.enter_mode("normal")
