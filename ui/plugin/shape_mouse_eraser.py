import numpy as np
from sapien.core import Pose
from .plugin_base import Plugin

from transforms3d import quaternions 
from pytorch3d.transforms import quaternion_apply
from tools.utils import totensor


class ShapeMouseEraser(Plugin):
    # currently there is only one shape
    mode_name = 'shape_mouse_eraser'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        self.next_shape = None
        
    def rendering(self):
        if self.viewer.window.key_press('e'):
            print('Shape Erasing')
            self._trigger()

    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self._next_shape = None
        self.viewer.toggle_pause(True)

    def get_object(self):
        return self.viewer.selected_softbody

    def leave_mode(self):
        state = self.engine._env.get_state()
        if state.X.size != 0:
            state.reid()
        self.update_scene_by_state(state)
        if self.viewer.paused:
            self.viewer.toggle_pause(self._old_paused)
        

    def step(self, action):
        return action, None

    def erase(self, dir, pos, iid):
        state = self.engine._env.get_state()
        if state.X[state.ids == iid].size == 0:
            return

        pos[[1, 2]] = pos[[2, 1]]
        dir[[1, 2]] = dir[[2, 1]]
        tx = state.X.copy() - pos
        dir = dir / np.sqrt((dir ** 2).sum().item())
        prj = (tx * tx).sum(axis=1) - (tx * dir).sum(axis=1) ** 2
        #print('camera dir and pos:', dir, pos)
        inn1 = (prj > 0.001) 
        inn2 = (state.ids != iid)
        inn = inn1|inn2

        state.X = state.X[inn]
        state.V = state.V[inn]
        state.F = state.F[inn]
        state.C = state.C[inn]
        state.ids = state.ids[inn]
        state.color = state.color[inn]
        state.E_nu_yield = state.E_nu_yield[inn]

        self.update_scene_by_state(state)

    def monitor(self):
        from .control_utils import normal_monitor
        normal_monitor(self.viewer)
        
        i = self.get_object()

        if i is not None:
            #TODO: need select entity in the end..

            import numpy as np
            from .control_utils import track_mouse
            window = self.viewer.window

            self.target = track_mouse(self, lambda: 0.0)

            if not self.target is None:
                self.erase(self.target[1], self.target[2], i['id'])
                
            if not self.viewer.paused:
                self.viewer.enter_mode("normal")

        if self.viewer.window.mouse_click(0):
            self.viewer.select(False)

