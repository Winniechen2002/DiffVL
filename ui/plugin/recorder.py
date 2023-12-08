import os
import torch
import tempfile
import datetime
from .plugin_base import Plugin
from sapien.core import renderer as R

class Recorder(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = None
        self.action = None

        self.record_screenshot = None
        self.images = []
        self.default_value=None
        self.to_save=None

    def imgui_ini(self):
        return 

    def build_windows(self):
        # from IPython import embed; embed()
        self.window = R.UIInputText().Label("save").Value("")
        return self.window#, #.Text("FPS: {:.2f} {}".format(self.window.fps, str(self.mode)))

    def dump_action(self):
        if self.action is not None and len(self.action) > 0:
            assert self.value is not None

            print("saving previous action", len(self.action), "to", self.value[:-3])
            torch.save(self.action, self.value[:-3] + '.action')


            self.action = None
            self.value = None

    def get_value(self):
        value = self.window.value.strip().replace("\x00", '')
        if len(value) == 0:
            value = self.default_value
        self.window.Value(" "*len(self.window.value)) # clear ..
        self.window.Size(0)
        self.window.Size(100)
        return value

    def dump_video(self):
        from tools.utils import animate
        print('saving to ..', self.video_address)
        animate(self.images[::10], self.video_address)
        self.images = None
        self.record_screenshot = None

    def rendering(self):
        if self.viewer.window.ctrl and self.viewer.window.key_press('a'):
            self.dump_action()
            raise NotImplementedError

        if self.viewer.window.ctrl and self.viewer.window.key_press('r'):
            print('start recording .. ')
            self.video_address = self.get_value() + ".mp4"
            self.record_screenshot = 0
            self.images = []


        if self.to_save is not None:
            # save at the next frame
            self.engine.save_scene_config(self.to_save)
            self.engine.refreshed = self.to_save
            self.to_save = None


        if self.viewer.window.ctrl and self.viewer.window.key_press('s'):
            if self.record_screenshot is not None and self.record_screenshot> 0:
                self.dump_video()
                return

            if self.action is not None and len(self.action) > 0:
                self.dump_action()
                return

            from frontend import DATA_PATH
            value = self.get_value()
            #self.gui.dump_action()
            print("trying to save to ...", value)
            if value is not None:
                self.engine._viewer.enter_mode('normal')
                self.to_save = value

        if self.record_screenshot is not None:
            import numpy as np
            self.record_screenshot += 1
            self.images.append(np.uint8(self.viewer.window.get_float_texture("Color") * 255))


    def step(self, action):
        if self.action is not None:
            self.action.append(action)

            if len(self.action) > 100000:
                self.dump_action()
        

        return action, None

    def close(self):
        self.window = None