from typing import Optional
from llm.tiny import Scene


class Plugin:
    mode_name = None

    def __init__(self, engine):
        from ..viewer import Viewer
        from ..gui import GUI

        self.viewer: Optional[Viewer] = None
        self.engine: GUI = engine

    def _trigger(self):
        assert self.mode_name is not None
        self.viewer.enter_mode(self.mode_name)

    def update_scene_by_state(self, state):
        self.engine.update_scene_by_state(state)
        # env = self.engine._env
        # scene = Scene(env, collect_obs=False)
        # env.set_state(state)
        # scene.collect(0)
        # scene.initial_state = state
        # self.engine.reload_scene(scene) # update based on scene ..

    def build_windows(self):
        return None

    def build_warning_windows(self):
        return None

    def enter_mode(self):
        pass
    
    def leave_mode(self):
        pass

    def rendering(self):
        # listen to keys or trigger the mode
        pass

    def monitor(self):
        # what would happen when endering the mode ..
        pass

    def step(self, cur_action=None):
        # modify simulation ..
        return cur_action, None

    def close(self):
        pass