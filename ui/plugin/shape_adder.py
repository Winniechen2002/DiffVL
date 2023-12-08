import os
import torch
import tempfile
import datetime
from .plugin_base import Plugin
from llm.tiny import Scene
from sapien.core import renderer as R
from envs.soft_utils import *
from envs.world_state import WorldState

class ShapeAdder(Plugin):
    mode_name = 'adder_shape'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = None
        self.action = None
        self.cube_length = 0.04
        self.ball_radius = 0.1
        self.rope_length = 0.1
        self.softness = 0
        self.cnt_objs = 0

        self.material_window = None
        self.E = -1.
        self.nu = -1.
        self.yi = -1.
        self.material_type = 'default'

        self.rect_height = 0.1
        self.rect_length = 0.1
        self.rect_width = 0.1

        self.cylinder_height = 0.02
        self.cylinder_radius = 0.1

        self.color_window = None
        self.color_type = 'Black'
        self.R = 0
        self.G = 0
        self.B = 0
        self.color_dict = {}
        try:
            COLOR_PATH = os.path.join(os.path.dirname(__file__), 'color_dict.txt')
            with open(COLOR_PATH, 'r') as f:
                for x in f:
                    x = x.split(' ')
                    self.color_dict[x[0]] = (int(x[1]),int(x[2]),int(x[3]))
        except IOError:
            print('Missing file color_dict.txt.')

        self.warning_window = None

    def imgui_ini(self):
        return 

    def build_windows(self):
        self.color_window = R.UIInputText().Value("").Label("color").Callback(lambda ct: self.collect_color(ct.value))
        self.material_window = R.UIInputText().Value("").Label("material").Callback(lambda m: self.collect_material(m.value))
        self.window = (
            R.UIWindow()
            .Pos(410, 10)
            .Size(200, 450)
            .Label("Add Shapes")
            .append(
                R.UIDisplayText().Text("Add a objects with color in 'color_dict.txt':"),
                self.color_window,
                R.UIDisplayText().Text("Add a objects with material [gold,iron,fiber,rubber,default]:"),
                self.material_window,
                R.UISameLine().append(
                    R.UIDisplayText().Text("Add a cube with length:"),
                    R.UIButton()
                    .Label("Add Cube")
                    .Callback(lambda pc: self.add_cube()),
                ),
                R.UISliderFloat()
                .Min(0.04)
                .Max(0.2)
                .Value(self.cube_length)
                .Label("Length")
                .Callback(lambda wc: self.collect_cube_length(wc.value)),
                R.UISameLine().append(
                    R.UIDisplayText().Text("Add a ball with diameter:"),
                    R.UIButton()
                    .Label("Add Ball")
                    .Callback(lambda pb: self.add_ball()),
                ),
                R.UISliderFloat()
                .Min(0.02)
                .Max(0.2)
                .Value(self.ball_radius)
                .Label("Radius")
                .Callback(lambda wb: self.collect_ball_radius(wb.value)),
                R.UISameLine().append(
                    R.UIDisplayText().Text("Add a rope with length:"),
                    R.UIButton()
                    .Label("Add Rope")
                    .Callback(lambda pp: self.add_rope()),
                ),
                R.UISliderFloat()
                .Min(0.1)
                .Max(0.8)
                .Value(self.rope_length)
                .Label("Longer Length")
                .Callback(lambda wp: self.collect_rope_length(wp.value)),
                R.UISameLine().append(
                    R.UIDisplayText().Text("Add a rectangle with [length,height,width]:"),
                    R.UIButton()
                    .Label("Add Rect")
                    .Callback(lambda pr: self.add_rect()),
                ),
                R.UISliderFloat()
                .Min(0.02)
                .Max(0.5)
                .Value(self.rect_height)
                .Label("height")
                .Callback(lambda rect_height: self.collect_rect_height(rect_height.value)),
                R.UISliderFloat()
                .Min(0.02)
                .Max(0.5)
                .Value(self.rect_length)
                .Label("length")
                .Callback(lambda rect_length: self.collect_rect_length(rect_length.value)),
                R.UISliderFloat()
                .Min(0.02)
                .Max(0.5)
                .Value(self.rect_width)
                .Label("width")
                .Callback(lambda rect_width: self.collect_rect_width(rect_width.value)),
                R.UISameLine().append(
                    R.UIDisplayText().Text("Add a cylinder with [height,radius]:"),
                    R.UIButton()
                    .Label("Add cylinder")
                    .Callback(lambda ac: self.add_cylinder()),
                ),
                R.UISliderFloat()
                .Min(0.01)
                .Max(0.8)
                .Value(self.cylinder_height)
                .Label("cylinder height")
                .Callback(lambda cyli_height: self.collect_cyli_height(cyli_height.value)),
                R.UISliderFloat()
                .Min(0.02)
                .Max(0.4)
                .Value(self.cylinder_radius)
                .Label("cylinder radius")
                .Callback(lambda cyli_radius: self.collect_cyli_radius(cyli_radius.value)),
                R.UIDisplayText().Text("WARNING: If you choose wrong parameter, the simulation will go wrong."),
                R.UIDisplayText().Text("Build a new material with E(Young's modulus), nu(Poisson's ratio) and yield strength:"),
                R.UIDisplayText().Text("Change Young's modulus"),
                R.UIInputFloat()
                .Value(self.E)
                .Label("Young's modulus (MPa)")
                .Callback(lambda E: self.collect_E(E.value)),
                R.UIDisplayText().Text("Change Poisson's ratio"),
                R.UIInputFloat()
                .Value(self.nu)
                .Label("Poisson's ratio")
                .Callback(lambda nu: self.collect_nu(nu.value)),
                R.UIDisplayText().Text("Change Yield strength"),
                R.UIInputFloat()
                .Value(self.yi)
                .Label("Yield strength")
                .Callback(lambda yi: self.collect_yi(yi.value)),
                
            )
        )
        return self.window #, #.Text("FPS: {:.2f} {}".format(self.window.fps, str(self.mode)))

    def collect_color(self, color_type):
        print('collecting color type')
        color_type = color_type.strip().replace("\x00", '')
        print(color_type)
        if color_type in self.color_dict:
            self.R, self.G, self.B = self.color_dict[color_type]
        else:
            self.warning_window = (
                R.UIWindow()
                .Pos(600, 400)
                .Size(200, 200)
                .Label("WARNING!")
                .append(
                    R.UIDisplayText().Text("This color is not supported!"),
                    R.UIButton()
                    .Label("Change the color!")
                    .Callback(lambda cww: self.close_warning_window()),
                )
            )
        self.color_type = color_type
        self.color_window.Size(0)
        self.color_window.Size(100)

    def collect_material(self, material_type):
        print('collecting material type')
        if material_type == 'rubber':
            self.E = 0.05*1000
            self.nu = 0.4999
            self.yi = 118.
        if material_type == 'gold':
            self.E = 77.2*1000
            self.nu = 0.43
            self.yi = 900.
        if material_type == 'fiber':
            self.E = 58.*1000
            self.nu = 0.3
            self.yi = 1800.
        if material_type == 'iron':
            self.E = 200.*1000
            self.nu = 0.3
            self.yi = 100.
        if material_type == 'default':
            self.E = -1.
            self.nu = -1.
            self.yi = -1.
        self.material_type = material_type
        self.material_window.Size(0)
        self.material_window.Size(100)

    def collect_E(self, E):
        print('collecting E')
        self.E = E

    def collect_nu(self, nu):
        print('collecting nu')
        self.nu = nu

    def collect_yi(self, yi):
        print('collecting yi')
        self.yi = yi

    def collect_cube_length(self, leng):
        print('collecting cube length')
        self.cube_length = leng
        # pass

    def collect_ball_radius(self, radi):
        print('collecting ball radius')
        self.ball_radius = radi
        # pass

    def collect_rect_length(self, lleng):
        print('collecting rect length')
        self.rect_length = lleng

    def collect_rect_height(self, hheight):
        print('collecting rect height')
        self.rect_height = hheight

    def collect_rect_width(self, wwidth):
        print('collecting rect width')
        self.rect_width = wwidth
        # pass

    def collect_rope_length(self, rleng):
        print('collecting rope length')
        self.rope_length = rleng

    def collect_cyli_height(self, cheight):
        print('collecting cylinder height')
        self.cylinder_height = cheight

    def collect_cyli_radius(self, cradius):
        print('collecting cylinder radius')
        self.cylinder_radius = cradius

    def close_warning_window(self):
        self.warning_window = None

    def build_warning_windows(self):
        return self.warning_window

    def add_shape(self, out):
        self._trigger()
        state = self.engine._env.get_state()
        # out = box(self.cube_length, center=(0.5, 0.5, 0.5), n = None)
        if len(state.X) + len(out) > 20000:
            self.warning_window = (
                R.UIWindow()
                .Pos(600, 400)
                .Size(200, 200)
                .Label("WARNING!")
                .append(
                    R.UIDisplayText().Text("UI can only build 20000 particles"),
                    R.UIButton()
                    .Label("Cancel the adding!")
                    .Callback(lambda cww: self.close_warning_window()),
                )
            )
            return None

        idxs = np.zeros(len(out)) + np.unique(state.ids).size

        E_nu_yield = np.zeros((len(out), 3))
        E_nu_yield[: , 0] = self.E
        E_nu_yield[: , 1] = self.nu
        E_nu_yield[: , 2] = self.yi
        new = WorldState.get_empty_state(n=len(out), E_nu_yield = E_nu_yield)
        new.X[:] = out
        new.ids[:] = idxs
        new.color[:] = rgb2int(self.R, self.G, self.B)
        state = state.add_state(new)
        self.update_scene_by_state(state)

    def add_cube(self):
        print('adding cube')
        out = box(self.cube_length, center=(0.5, 0.5, 0.5), n = None)
        self.add_shape(out)
        print('finished adding cube {} with color {}'.format(self.cube_length,self.color_type))

    def add_ball(self):
        print('adding ball')
        out = sphere(self.ball_radius, center=(0.5, 0.5, 0.5), n = None)
        self.add_shape(out)
        print('finished adding ball {} with color {}'.format(self.ball_radius,self.color_type))

    def add_rect(self):
        print('adding rect')
        out = box(width=[self.rect_width, self.rect_height, self.rect_length], center=(0.5, 0.5, 0.5), n = None)
        self.add_shape(out)
        print('finished adding rect[height,width,length] {} with color {}'.format((self.rect_height, self.rect_width, self.rect_length),self.color_type))

    def add_rope(self):
        print('adding rope')
        out = box(width=[0.02, 0.02, self.rope_length], center=(0.5, 0.5, 0.5), n = None)
        self.add_shape(out)
        print('finished adding rope {} with color {}'.format(self.rope_length,self.color_type))

    def add_cylinder(self):
        print('adding cylinder')
        out = cylinder(args=[self.cylinder_radius, self.cylinder_height], center=(0.5, 0.5, 0.5), n = None)
        self.add_shape(out)
        print('finished adding cylinder[radius,height] {} with color {}'.format((self.cylinder_radius, self.cylinder_height),self.color_type))

    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self.viewer.toggle_pause(True)

    def leave_mode(self):
        if self.viewer.paused:
            self.viewer.toggle_pause(self._old_paused)

    def step(self, action):
        return action, None

    def monitor(self):
        if not self.viewer.paused:
            self.viewer.enter_mode("normal")

    def close(self):
        self.window = None