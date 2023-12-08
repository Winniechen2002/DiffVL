import numpy as np
from sapien.core import Pose
from .plugin_base import Plugin

from transforms3d import quaternions 
from pytorch3d.transforms import quaternion_apply
from tools.utils import totensor


class ShapeDeleter(Plugin):
    # currently there is only one shape
    mode_name = 'delete_shape'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        
    def rendering(self):
        if self.viewer.window.key_press('r'):# and self.selected:
            print('Shape deleting')
            self._trigger()

    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self.viewer.toggle_pause(True)

    def get_object(self):
        #for i in self.engine._soft_bodies.values():
        #    return i
        return self.viewer.selected_softbody

    def leave_mode(self):
        if self.viewer.paused:
            # recover ..
            self.viewer.toggle_pause(self._old_paused)

    def step(self, action):
        return action, None

    def monitor(self):
        from .control_utils import normal_monitor
        normal_monitor(self.viewer)
        
        i = self.get_object()


        if i is not None:
            #TODO: need select entity in the end..
            # print(i)

            if self.viewer.window.key_press('d'):

                state = self.engine._env.get_state()
                if state.X.size == 0:
                    return

                inn = state.ids != i['id']
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
                
            if not self.viewer.paused:
                self.viewer.enter_mode("normal")

                
        if self.viewer.window.mouse_click(0):
            self.viewer.select(False)
