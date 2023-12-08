'''
self.depth is the depth of the plane.
Now the points with same depth are on a sphere surface around the camera, I hope to modify it into all the points with same
depth are on a normal plane of camera ray.
Now one click may lead to infinite velocity because the change of mechanism of target.
'''
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
from .control_utils import control3d, track_mouse_mover, camera_point_to_plane_dir

from llm.tiny.softbody import SoftBody
from sapien.core import renderer as R


class SelectDragger(Plugin):
    # currently there is only one shape
    mode_name = 'select_dragger'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        self.next_shape = None
        self.las_pos = None
        self.id = None
        self.N = None
        self.range = 0.015
        self.eps = 1e-5
        self.sensitivity = 8
        self.wheel_sensitivity = 0.25
        self.color_copy = None
        self.target = None
        self.ray = None

    def collect_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity

    def collect_wheel_sensitivity(self, wheel_sensitivity):
        self.wheel_sensitivity = wheel_sensitivity

    def collect_range(self, range):
        self.range = range

    def build_windows(self):
        self.window = (
            R.UIWindow()
            .Pos(410, 10)
            .Size(200, 300)
            .Label("Select Drag")
            .append(
                R.UIDisplayText().Text("Sensitivity of dragger:"),
                R.UISliderFloat()
                .Min(1)
                .Max(20)
                .Value(self.sensitivity)
                .Label("Sensitivity")
                .Callback(lambda wc: self.collect_sensitivity(wc.value)),
                R.UIDisplayText().Text("Sensitivity of wheel:"),
                R.UISliderFloat()
                .Min(0.05)
                .Max(0.8)
                .Value(self.wheel_sensitivity)
                .Label("Wheel Sensitivity")
                .Callback(lambda wc: self.collect_wheel_sensitivity(wc.value)),
                R.UIDisplayText().Text("Range of dragger:"),
                R.UISliderFloat()
                .Min(0.01)
                .Max(0.5)
                .Value(self.range)
                .Label("Range")
                .Callback(lambda wc: self.collect_range(wc.value)),
            )
        )
        return self.window

    def change_opacity(self, x, opacity):
        '''
                There are still some bugs here, this piece of code only change the opacity of the selected parts of 
            self.viewer.selected_softbody.
                You will also move the selected parts of nearby softbodies but the opacity of them won't change.
        '''
        if x == None:
            return
        tmp_dict = {}
        for i in self.id:
            tmp_dict[i] = opacity
        # print("ids ",self.id)
        
        for i in range(len(x['indices'])):
            ind = int(x['indices'][i])
            x['tensor'][i,7] = tmp_dict.get(ind,0)


    def rendering(self):
        if self.viewer.window.key_press('n'):# and self.selected:
            print('select dragger')
            self._trigger()

    def leave_mode(self):
        # print("leave",self.color_copy)
        if self.viewer.paused:
            self.viewer.toggle_pause(self._old_paused)

    def get_itsc(self):
        from .control_utils import track_mouse_mover3d
        self.viewer.select()
        self.las_pos, ray, pos = track_mouse_mover3d(self)
        stt = self.engine._env.get_state()
        # print(stt.color)
        N = stt.X.shape[0]
        
        pos[[1, 2]] = pos[[2, 1]]
        ray[[1, 2]] = ray[[2, 1]]
        print("pos",pos, "las_pos",self.las_pos)
        
        tx = stt.X - pos
        
        dir = ray / math.sqrt((ray ** 2).sum().item())
        self.ray = dir

        prj = (tx * dir).sum(axis=1)
        dis = (tx * tx).sum(axis=1) - prj ** 2
        tmp = np.where(dis < self.eps)
        if len(tmp[0]) == 0:
            self.leave_mode()
            return tmp[0], 0
        minn = tmp[0][np.argmin(prj[tmp[0]])]
        first_point = stt.X[minn]
        print(first_point,pos,self.ray)
        self.depth = ((first_point-pos) * self.ray).sum()
        print("depth:",self.depth)
        dis = ((stt.X-first_point) * (stt.X-first_point)).sum(axis=1)
        tmp = np.where(dis < self.range)
        return tmp[0], N


    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self._next_shape = None

    def get_object(self):
        return self.viewer.selected_softbody

    def step(self, action):
        if self.target is not None:

            self.las_pos = self.engine.last_scene.initial_state.X[self.id[0]]

            dir = np.array(self.target-self.las_pos)
            srt = math.sqrt((dir**2).sum().item())
            if srt != 0:
                dir = dir / srt * 0.1

            state = self.engine._env.get_state()
            state.V[self.id] = dir * self.sensitivity
            self.engine._env.set_state(state)

        return action, None

    def generate_target(self):
        window = self.viewer.window
        mx, my = np.array(window.mouse_position)
        if self.viewer.is_mouse_available(mx, my):
            ww, wh = window.size
            mx = (mx / ww - 0.5)*2
            my = (my / wh - 0.5)*2
            print("mx",mx,"my",my)
            dir = camera_point_to_plane_dir(self, mx, my)
            dir = dir / math.sqrt((dir ** 2).sum().item())
            print("dir",dir,(dir * self.ray).sum())
            pos = window.get_camera_position()
            print("rotate:",window.get_camera_rotation())
            coord = (self.depth / (dir * self.ray).sum()) * dir + pos
            return coord
        return None

    def monitor(self):
        from .control_utils import track_mouse_pos
        from envs.soft_utils import sphere

        self.id, self.N = self.get_itsc()
        # print("render",self.id,len(self.id))
        if len(self.id) > 0:
            # stt = self.engine._env.get_state()
            # print("stt:",stt)
            # self.color_copy = stt.color[self.id]
            # print("copy",self.color_copy)
            # stt.color[self.id] = rgb2int(255,255,255)
            # self.update_scene_by_state(stt)
            self.change_opacity(self.viewer.selected_softbody,0.3)
            wx, wy = self.viewer.window.mouse_wheel_delta
            if wx != 0:
                print("wheel: ",wx)
                self.depth += wx * self.wheel_sensitivity

            if self.viewer.window.mouse_click(1):
                print('click2')
            
                self.target = self.generate_target()
                print("target:",self.target)

            if not self.viewer.paused:
                self.viewer.enter_mode("normal")
            
        
        if self.viewer.window.mouse_click(0):
            self.viewer.select(False)
            if self.N and len(self.id) > 0:
                self.change_opacity(self.viewer.selected_softbody,0)
            state = self.engine._env.get_state()
            state.V[self.id] = 0
            self.engine._env.set_state(state)
            self.N = None
            self.id = None
            self.target = None
            # self.leave_mode()
            # self.viewer.enter_mode("normal")
