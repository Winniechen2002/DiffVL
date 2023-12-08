from .plugin_base import Plugin

class Switcher(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rendering(self):
        update = None
        for i in '12345':
            if self.viewer.window.key_down(i):
                update = int(i)

        if update is not None:
            state = self.engine._env.get_state()
            state = state.switch_tools(list(self.engine._env.tools.keys())[update-1])
            self.engine._env.set_state(state)

            self.engine._add_sapien_tools(state.tool_name)


