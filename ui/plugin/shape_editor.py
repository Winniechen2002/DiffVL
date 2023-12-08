import numpy as np
from sapien.core import Pose
from .plugin_base import Plugin

from transforms3d import quaternions 
from pytorch3d.transforms import quaternion_apply
from tools.utils import totensor
from .control_utils import control3d


class ShapeMover(Plugin):
    # currently there is only one shape
    mode_name = 'editor_shape'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        self.next_shape = None
        
    def rendering(self):
        if self.viewer.window.key_press('m'):# and self.selected:
            self._trigger()

    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self._next_shape = None
        self.viewer.toggle_pause(True)

    def get_object(self):
        #for i in self.engine._soft_bodies.values():
        #    return i
        return self.viewer.selected_softbody

    def leave_mode(self):
        if self.viewer.paused:
            # recover ..
            self.viewer.toggle_pause(self._old_paused)
        
        #obj = self.get_object()
        if self.update is not None:
            self.changes = {}
            for i, k in self.engine._soft_bodies.items():
                self.changes[i] = k['tensor'][:, :3].detach().cpu().numpy()

    def step(self, action):
        if self.update is not None:
            # raise NotImplementedError
            state = self.engine._env.get_state()
            for i, k in self.engine._soft_bodies.items():
                state.X[k['indices'].detach().cpu().numpy()] = self.changes[i][:, [0, 2, 1]]
            self.engine._env.set_state(state)

            self.next_shape = None
            self.update = None

        return action, None

    def monitor(self):
        from .control_utils import normal_monitor
        normal_monitor(self.viewer)

        i = self.get_object()

        if i is not None:
            action = control3d(self)
            #TODO: need select entity in the end..
            print("action",action)
            if np.linalg.norm(action) > 0:
                self.update = True

                com = i['tensor'][:, :3].mean(axis=0)
                from .control_utils import plane_xy
                basis = plane_xy(self, com.detach().cpu().numpy())
                action[:2] = action[:2] @ basis

                p = action[:3] * 0.02
                q = quaternions.axangle2quat([0, 0., 1.], 0.02 * action[3])


                i['tensor'][:, :3] = quaternion_apply(
                    totensor(q, 'cuda:0'), i['tensor'][:, :3] - com 
                ) + com + totensor(p, 'cuda:0')

                
            if not self.viewer.paused:
                self.viewer.enter_mode("normal")


        if self.viewer.window.mouse_click(0):
            self.viewer.select(False)
