from .plugin_base import Plugin
import numpy as np
import torch

class MouseEraser(Plugin):
    mode_name = 'mouse_erase'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        self.next_shape = None
        # self.leave_mode()

    def add_display_circle(self):
        rs = self.viewer.scene.renderer_scene
        render_scene = rs._internel_scene
        if self.viewer.display_object:
            render_scene.remove_node(self.display_object)
        self.display_object = render_scene.add_node()
        s2w = self.viewer.window.mouse_position

    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self._next_shape = None
        self.viewer.toggle_pause(True)
        
        
    def leave_mode(self):
        self.wx = None
        self.target = None
        self.up = None
        self.left = None
        self.front = None

        if self.viewer.paused:
            self.viewer.toggle_pause(self._old_paused)
       
    def rendering(self):
        viewer = self.viewer
        if viewer.window.key_press('y'): # and self.selected:
            print('mouse erasing')
            self._trigger()

    def erase(self, dir, pos):
        state = self.engine._env.get_state()
        if state.X.size == 0:
            return

        pos[[1, 2]] = pos[[2, 1]]
        dir[[1, 2]] = dir[[2, 1]]
        tx = state.X.copy() - pos
        dir = dir / np.sqrt((dir ** 2).sum().item())
        prj = (tx * tx).sum(axis=1) - (tx * dir).sum(axis=1) ** 2
        #print('camera dir and pos:', dir, pos)
        inn = prj > 0.01

        state.X = state.X[inn]
        state.V = state.V[inn]
        state.F = state.F[inn]
        state.C = state.C[inn]
        state.ids = state.ids[inn]
        state.color = state.color[inn]
        state.E_nu_yield = state.E_nu_yield[inn]

        if state.X.size != 0: 
            state.reid()
        self.update_scene_by_state(state)

    def monitor(self):
        import numpy as np
        from .control_utils import track_mouse
        window = self.viewer.window

        self.target = track_mouse(self, lambda: float(self.engine.obs['qpos'][1]))
        # print (self.target)

        if not self.target is None:
            # print(self.target)
            self.erase(self.target[1], self.target[2])


        if window.mouse_click(0):
            self.viewer.enter_mode("normal")

    def step(self, cur_action=None):
        return cur_action, None