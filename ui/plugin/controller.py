# normal controller 
from .plugin_base import Plugin
import numpy as np


class NormalController(Plugin):
    def __init__(self, engine):
        super().__init__(engine)
        self.action = None

    def rendering(self):
        if self.engine.selected_tool() and self.viewer.mode == 'normal':
            if self.engine.tool_name == 'Gripper': 
                action = np.zeros(7)
                for key in ['w', 's', 'a', 'd']:
                    if self.viewer.window.key_press(key):
                        self.viewer.enter_mode('normal')
                        if key == 'w':
                            action[1] += 0.5
                        elif key == 's':
                            action[1] -= 0.5
                        elif key == 'd':
                            action[0] += 0.5
                        else:
                            action[0] -= 0.5
                if np.linalg.norm(action) > 0.:
                    action[[1, 2]] = action[[2, 1]]
                    self.action = action
                    self.action_count = 1

            elif self.engine.tool_name == 'DoublePushers':
                from .control_utils import control6d
                action = control6d(self)
                if np.linalg.norm(action) > 0.:
                    action = np.concatenate([action, action])
                    self.action = action
                    self.action_count = 1

            elif self.engine.tool_name == 'Pusher':
                from .control_utils import control6d, control3d
                action = control6d(self)
                if np.linalg.norm(action) > 0.:
                    # action = np.concatenate([action, action])
                    self.action = action
                    self.action_count = 1

            elif self.engine.tool_name == 'Knife':
                from .control_utils import control6d, control3d
                action = control6d(self)
                if np.linalg.norm(action) > 0.:
                    self.action = action
                    self.action_count = 1
            elif self.engine.tool_name == 'Rolling_Pin':
                from .control_utils import control6d, control3d
                action = control6d(self)
                if np.linalg.norm(action) > 0.:
                    # action = np.concatenate([action, action])
                    self.action = action
                    self.action_count = 1
        else:
            self.action = None

    def step(self, action=None):
        if self.action is not None:
            action = self.action

            self.action_count -= 1
            if self.action_count <= 0:
                self.action = None

        return action, None