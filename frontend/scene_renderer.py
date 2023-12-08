import os
import numpy as np 
import torch
import pickle
from .scene_api import *
from ui import GUI as GUIBase


class GUI(GUIBase):
    def __init__(self, cfg=None):
        super().__init__()
        self.refreshed=None
        self.config = None

    def take_picture(self):
        return np.uint8(self._viewer.window.get_float_texture("Color") * 255)

    def capture_segments(self, states, names):
        assert len(names) < 20 # less than 10 elements
        values = np.zeros(states.X.shape[0], dtype=np.int32)

        for idx, (k, v) in enumerate(names.items()):
            if 'id' in v:
                values[v['id']] |= (1 << idx)

        previous_labels = {}
        for k, v in self._soft_bodies.items():
            previous_labels[k] = v['tensor'][:].detach().clone()
            #print(k, v['tensor'].shape, v['id'].shape)
            #print(v['id'])
            id = int(v['id'])
            mask = states.ids == id 
            # 9, 10, 11
            v['tensor'][:, 0+9] = torch.tensor(values[mask].astype(np.float32), dtype=torch.float32, device=v['tensor'].device)

        out = self._viewer.window.get_uint32_texture("Segmentation")[..., 2]
        for k, v in self._soft_bodies.items():
            v['tensor'][:] = previous_labels[k]
        return out

    def load_scene_config(self, config: SceneConfig, id=None):
        #self._scene.load_scene_config(config)
        #raise NotImplementedError
        self.update_scene_by_state(config.state)
        for i in self._viewer.plugins:
            if hasattr(i, 'default_value'):
                #i.load_scene_config(config, id)
                i.default_value = id
        self.config = config

    def get_scene_config(self):
        state=self._env.get_state()
        # if self.config is None:
        #     config = SceneConfig(state)
        # else:
        #     print('set as current state ..')
        #     config = self.config
        #     config.state = state
        config = SceneConfig(state)

        segments = self.capture_segments(config.state, config.names)
        config.views = {
            'screen': self.take_picture(),
            'seg': segments
        }
        self.config = None
        return config


    def save_scene_config(self, name=None):
        out = self.get_scene_config()

        if name is not None:
            from . import DATA_PATH
            name = os.path.join(DATA_PATH, 'scene_' + str(name))
            print('save at', name)
            with open(name + '.pkl', 'wb') as f:
                pickle.dump(out, f)

        return out