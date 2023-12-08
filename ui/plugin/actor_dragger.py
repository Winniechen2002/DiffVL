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


class ActorDragger(Plugin):
    # currently there is only one shape
    mode_name = 'Actor Dragger'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ids = []
        self.targets = []
        self.actor = None
        self.actor_pos = None
        self.idx = None

        self.range = 0.015
        self.eps = 1e-5
        self.sensitivity = 0.8
        self.pause = False
        self.window = None

    def build_windows(self):
        self.window = (
            R.UIWindow()
            .Label("Actor Drag")
            .append(
                R.UIDisplayText().Text("Sensitivity of dragger:"),
                R.UISliderFloat()
                .Min(0.1)
                .Max(10)
                .Value(self.sensitivity)
                .Label("Sensitivity")
                .Callback(lambda wc: self.collect_sensitivity(wc.value)),
                R.UIDisplayText().Text("Range of dragger:"),
                R.UISliderFloat()
                .Min(0.001)
                .Max(0.2)
                .Value(self.range)
                .Label("Range")
                .Callback(lambda wc: self.collect_range(wc.value)),
            )
        )
        return self.window

    def collect_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity

    def collect_range(self, range):
        self.range = range

    def set_opacity(self, x, opacity = 0.3, idx = []):
        if x == None:
            return
        tmp_dict = {}
        for i in idx:
            tmp_dict[i] = opacity
        
        for i in range(len(x['indices'])):
            ind = int(x['indices'][i])
            x['tensor'][i,7] = tmp_dict.get(ind,0)

    def build_actor(self):
        builder = self.engine._scene.create_actor_builder()
        material = self.engine._renderer.create_material()
        material.base_color = [1., 1., 1., 5.]
        material.roughness = 1
        builder.add_sphere_visual(radius=0.02, material=material)
        self.actor = builder.build_static()
        self.actor_pos = [0.5, 0.5, 0.5]
        self.actor.set_pose(Pose(p=self.actor_pos))

    def remove_actor(self):
        self.engine._scene.remove_actor(self.actor)

    def rendering(self):
        if self.viewer.window.key_press('b'):# and self.selected:
            print('actor dragger')
            self._trigger()

    def leave_mode(self):
        self.ids = []
        self.targets = []
        self.idx = None
        self.remove_actor()
        # print("leave",self.color_copy)
        if self.viewer.paused:
            self.viewer.toggle_pause(self._old_paused)

    def get_itsc(self):
        from .control_utils import track_mouse_mover3d
        _, ray, pos = track_mouse_mover(self)
        stt = self.engine._env.get_state()
        
        pos[[1, 2]] = pos[[2, 1]]
        ray[[1, 2]] = ray[[2, 1]]
        
        tx = stt.X - pos
        
        dir = ray / math.sqrt((ray ** 2).sum().item())
        
        prj = (tx * dir).sum(axis=1)
        dis = (tx * tx).sum(axis=1) - prj ** 2
        tmp = np.where(dis < self.eps)
        if len(tmp[0]) == 0:
            return None
        
        minn = tmp[0][np.argmin(prj[tmp[0]])]
        first_point = stt.X[minn]
        
        dis = ((stt.X-first_point) * (stt.X-first_point)).sum(axis=1)
        tmp = np.where(dis < self.range)
        return tmp[0]


    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self.build_actor()

    def get_object(self):
        return self.viewer.selected_softbody

    def step(self, action):
        return action, None

    def monitor(self):
        from .control_utils import track_mouse_pos
        from .control_utils import normal_monitor
        from envs.soft_utils import sphere

        normal_monitor(self.viewer)
        if self.viewer.window.key_press('x'):#confilct with shape_delete
            self.ids = []
            self.targets = []
            self.idx = None
            if self.viewer.selected_softbody:
                self.set_opacity(self.viewer.selected_softbody)

        if self.viewer.window.key_press('v'):
            self.pause = True
        if self.viewer.window.key_press('c'):
            self.pause = False

        
        if self.viewer.window.mouse_click(0):
            self.viewer.select(False)
            self.idx = self.get_itsc()
            if self.idx is not None:
                self.set_opacity(self.viewer.selected_softbody, 0.3, self.idx)


        if self.viewer.window.key_press('w'):
            self.actor_pos[2] += 0.01

        if self.viewer.window.key_press('s'):
            self.actor_pos[2] -= 0.01

        if self.viewer.window.key_press('d'):
            self.actor_pos[0] += 0.01

        if self.viewer.window.key_press('a'):
            self.actor_pos[0] -= 0.01

        if self.viewer.window.key_press('i'):
            self.actor_pos[1] += 0.01

        if self.viewer.window.key_press('k'):
            self.actor_pos[1] -= 0.01

        self.actor.set_pose(Pose(p=[self.actor_pos[0],self.actor_pos[2],self.actor_pos[1]]))
            
        if self.viewer.selected_softbody:
            if self.idx is not None:
                if self.viewer.window.key_press('z'):
                    print('Add force with target pos:{}'.format(self.actor_pos))
                    self.targets.append(self.actor_pos.copy())
                    self.ids.append(self.idx.copy())
                    self.idx = None
                    self.set_opacity(self.viewer.selected_softbody)

        if not self.pause:
            if self.targets is not None:
                state = self.engine._env.get_state()
                for idx, target in zip(self.ids, self.targets):
                    pos = state.X[idx]
                    # pos = pos[:,[0,2,1]]
                    # target = [target[0],target[2],target[1]]
                    dir = target - pos 
                    # print(pos, target , dir)
                    # srt = math.sqrt((dir**2).sum().item())
                    # if srt != 0:
                        # dir = dir / srt
                    # dir = dir[:,[0,2,1]] 
                    state.V[idx] = dir * self.sensitivity
                self.engine._env.set_state(state)
        else:
            state = self.engine._env.get_state()
            state.V[:] = 0
            self.engine._env.set_state(state)
