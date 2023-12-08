import os
import torch
import tempfile
import datetime
from .plugin_base import Plugin
from llm.tiny import Scene
from sapien.core import renderer as R
from envs.soft_utils import *
from envs.world_state import WorldState

#bugs here
class MouseSelector(Plugin):
    mode_name = "mouse_selector"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = None
        self.actor = None
        self.update = None
        self.next_shape = None
    
    def rendering(self):
        if self.viewer.window.key_press('n'):
            self._trigger()
    
    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self._next_shape = None
        self.viewer.toggle_pause(True)

    def leave_mode(self):
        if self.viewer.paused:
            self.viewer.toggle_pause(self._old_paused)

    def step(self, action):
        return action, None

    def get_softbody(self):
        return self.viewer.selected_softbody

    def get_entity(self):
        return self.viewer.selected_entity

    def build_windows(self):
        softbody = self.get_softbody()
        self.window = (
            R.UIWindow()
            .Pos(310, 10)
            .Size(200, 200)
            .Label("Selected Point")
            .append(
                R.UIDisplayText().Text("Type: "),
                R.UIDisplayText().Text("id: "),
                R.UIDisplayText().Text("Color: "),
                R.UIDisplayText().Text("paricle numbers: ")
            )
        )
        if softbody:
            self.window.get_children[1].Text("id: {}".format(softbody[id]))
        return self.window
    
    def monitor(self):
        if not self.viewer.paused:
            self.viewer.enter_mode("normal")

    def close(self):
        self.window = None