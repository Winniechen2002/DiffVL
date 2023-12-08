from .plugin_base import Plugin
import numpy as np


class MouseTracker(Plugin):
    mode_name = 'mouse_track'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leave_mode()

    def enter_mode(self):
        viewer = self.viewer
    
    def leave_mode(self):
        self.wx = None
        self.target = None
        self.up = None
        self.rot = None

    def rendering(self):
        # listen to keys or trigger the mode
        #print(self.engine.tool_name)
        if self.engine.tool_name == 'Gripper':
            viewer = self.viewer
            #self.selected = self.engine.selected_tool()
            if viewer.window.key_press('t'):# and self.selected:
                print('tracking')
                self._trigger()

    def monitor(self):
        # what would happen when endering the mode ..
        import numpy as np
        from .control_utils import track_mouse
        window = self.viewer.window

        self.target = track_mouse(self, lambda: float(self.engine.obs['qpos'][1]))
        wx, wy = window.mouse_wheel_delta
        #print(wx, wy)
        if wx != 0:
            self.wx = wx
        else:
            self.wx = None

        self.up = 0
        if self.viewer.window.key_press('w'):
            self.up = 1
        if self.viewer.window.key_press('s'):
            self.up = -1

        self.rot = 0
        if self.viewer.window.key_press('a'):
            self.rot = -1
        if self.viewer.window.key_press('d'):
            self.rot = 1

        if window.mouse_click(0):
            self.viewer.enter_mode("normal")
        # print(self.wx, self.up, self.rot)

    def step(self, cur_action=None):
        # modify simulation ..

        if self.target is not None:
            qpos = self.engine.obs['qpos'].detach().cpu().numpy()
            cur_action[[0, 2]] = (self.target[0] - qpos[[0, 2]]).clip(-0.5, 0.5)

        if self.wx is not None:
            cur_action[-1] = -max(min(self.wx / 5, 0.5), -0.5)
            #cur_action = cur_action

        if self.up is not None:
            cur_action[1] = self.up * 0.3
        if self.rot is not None:
            cur_action[4] = self.rot

        return cur_action, None