import numpy as np
from sapien.core import Pose
from .plugin_base import Plugin

from transforms3d import quaternions 
from pytorch3d.transforms import quaternion_apply
from tools.utils import totensor
from envs.soft_utils import *
from sapien.core import renderer as R
from envs.world_state import WorldState


class Reshaper(Plugin):
    # currently there is only one shape
    mode_name = 'Reshaper'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        self.window = None
        self.action = None
        self.object = None
        self.rect_height = 0.1
        self.rect_length = 0.1

        self.cylinder_radius = 0.02
    
    def imgui_ini(self):
        return 

    def build_windows(self):
        self.window = R.UIWindow().Label("Reshaper")
        # from .plugin import Reshaper
        self.window.append(
                R.UIDisplayText().Text("Change the object to cube:"),
                R.UIButton()
                .Label("Cube")
                .Callback(lambda tc: self.to_cube()),
                R.UIDisplayText().Text("Change the object to ball:"),
                R.UIButton()
                .Label("Ball")
                .Callback(lambda tb: self.to_ball()),
                R.UIDisplayText().Text("Change the object to rect with [length,height]:"),
                R.UISliderFloat()
                .Min(0.04)
                .Max(0.3)
                .Value(self.rect_height)
                .Label("height")
                .Callback(lambda rect_height: self.collect_rect_height(rect_height.value)),
                R.UISliderFloat()
                .Min(0.04)
                .Max(0.3)
                .Value(self.rect_length)
                .Label("length")
                .Callback(lambda rect_length: self.collect_rect_length(rect_length.value)),
                R.UIButton()
                .Label("Rect")
                .Callback(lambda tr: self.to_rect()),
                R.UIDisplayText().Text("Change the object to rect with radius:"),
                R.UISliderFloat()
                .Min(0.02)
                .Max(0.4)
                .Value(self.cylinder_radius)
                .Label("cylinder radius")
                .Callback(lambda cyli_radius: self.collect_cyli_radius(cyli_radius.value)),
                R.UIButton()
                .Label("Cylinder")
                .Callback(lambda tr: self.to_cyli()),
            )
        return self.window #, #.Text("FPS: {:.2f} {}".format(self.window.fps, str(self.mode)))

    def get_object(self):
        return self.viewer.selected_softbody

    def to_shape(self, out, i):
        self._trigger()

        state = self.engine._env.get_state()
        if state.X.size == 0:
            return

        inn = state.ids == i['id']
        state.X[inn] = out

        self.update_scene_by_state(state)

    def to_cube(self):
        print('changing into cube')
        i = self.get_object()
        if i is not None:
            N = i['N']
            volume = compute_n_volume(N)
            out = box(volume**(1/3), center=(0.5, 0.5, 0.5), n = N)
            self.to_shape(out, i)
            print('finished changing to cube')

    def to_ball(self):
        print('changing into ball')
        i = self.get_object()
        if i is not None:
            N = i['N']
            volume = compute_n_volume(N)
            out = sphere((volume * 3 / (4 * np.pi))**(1/3), center=(0.5, 0.5, 0.5), n = N)
            self.to_shape(out, i)
            print('finished changing to ball')


## rect

    def collect_rect_length(self, lleng):
        print('collecting rect length')
        self.rect_length = lleng

    def collect_rect_height(self, hheight):
        print('collecting rect height')
        self.rect_height = hheight

    def to_rect(self):
        print('changing into rect')
        i = self.get_object()
        if i is not None:
            N = i['N']
            rect_width = compute_n_volume(N) / (self.rect_length * self.rect_height)
            out = box(width=[rect_width, self.rect_height, self.rect_length], center=(0.5, 0.5, 0.5), n = N)
            self.to_shape(out, i)
            print('finished changing to rect with [height,width,length]:'.format([self.rect_height, rect_width, self.rect_length]))

    def collect_cyli_radius(self, cradius):
        print('collecting cylinder cradius')
        self.cylinder_radius = cradius

    def to_cyli(self):
        print('changing into cylinder')
        i = self.get_object()
        if i is not None:
            N = i['N']
            cylinder_height = compute_n_volume(N) / (np.pi * self.cylinder_radius * self.cylinder_radius)
            out = cylinder(args=[self.cylinder_radius, cylinder_height], center=(0.5, 0.5, 0.5), n = N)
            self.to_shape(out, i)
            print('finished changing to cylinder with [radius,height] {}'.format((self.cylinder_radius, cylinder_height)))



    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self.viewer.toggle_pause(True)

    def leave_mode(self):
        self.object = None
        if self.viewer.paused:
            self.viewer.toggle_pause(self._old_paused)

    def step(self, action):
        return action, None

    def monitor(self):
        if not self.viewer.paused:
            self.viewer.enter_mode("normal")

    def close(self):
        self.window = None
