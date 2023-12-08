import numpy as np
# from envs.world_state import WorldState
from .plugin_base import Plugin

from transforms3d import quaternions 
from .control_utils import control3d

from llm.tiny.softbody import SoftBody


class ShapeDrawer(Plugin):
    # currently there is only one shape
    mode_name = 'draw_shape'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        self.next_shape = None
        
    def rendering(self):
        if self.viewer.window.key_press('p'):# and self.selected:
            self._trigger()

    def enter_mode(self):
        self.shape = self.get_object().copy()
        stt = self.engine.last_scene.env.initial_state
        inds = np.where(stt.ids == self.shape['id'])[0]
        
        stt.X = np.concatenate((stt.X[inds], np.delete(stt.X, inds, axis=0)), axis=0)
        stt.V = np.concatenate((stt.V[inds], np.delete(stt.V, inds, axis=0)), axis=0)
        stt.F = np.concatenate((stt.F[inds], np.delete(stt.F, inds, axis=0)), axis=0)
        stt.C = np.concatenate((stt.C[inds], np.delete(stt.C, inds, axis=0)), axis=0)
        stt.ids = np.concatenate((stt.ids[inds], np.delete(stt.ids, inds, axis=0)), axis=0)
        stt.color = np.concatenate((stt.color[inds], np.delete(stt.color, inds, axis=0)), axis=0)
        stt.E_nu_yield = np.concatenate((stt.E_nu_yield[inds], np.delete(stt.E_nu_yield, inds, axis=0)), axis=0)
        self.engine.last_scene.env.initial_state = stt
        self.en = stt.ids.size
        
        self.cur_shape = self.get_object()
        self._old_paused = self.viewer.paused
        self._next_shape = None
        self.viewer.toggle_pause(True)

    def get_object(self):
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

        if self.cur_shape is not None:
            action = control3d(self)
            #TODO: need select entity in the end..

            if np.linalg.norm(action) > 0:
                self.update = True

                com = self.cur_shape['tensor'][:, :3].mean(axis=0)
                from .control_utils import plane_xy
                basis = plane_xy(self, com.detach().cpu().numpy())
                action[:2] = action[:2] @ basis

                p = action[:3] * 0.02
                q = quaternions.axangle2quat([0, 0., 1.], 0.02 * action[3])

                N = self.cur_shape['N']
                # print(N)
                scn = self.engine.last_scene
                stt = scn.env.initial_state
                stt.add_part(self.shape['id'], self.shape['N'], self.en, com, p, q)              
                scn.initial_state = scn.env.initial_state = stt
                scn.env.set_state(stt)        
                self.engine.reload_scene(scn)

                
            if not self.viewer.paused:
                self.viewer.enter_mode("normal")


        # print('end')
        if self.viewer.window.mouse_click(0):
            print('click')
            self.viewer.select(False)
            self.leave_mode()
            self.viewer.enter_mode("normal")
