import numpy as np
from envs.world_state import WorldState

class SceneConfig:
    # object to communicated between GUI and the renderer
    # what we need to do is to assign name to the object.
    def __init__(
        self,
        state: WorldState,
        names=None,
        tokens=None,
        views=None, # for each view, return its image and capture
        kwargs=None,
    ):
        self.state = state

        if names is None:
            names = {}
            max_idx = int(np.unique(state.ids)[0] + 2)
            for i in range(max_idx):
                mask = (state.ids == i)
                print(i, mask.sum())
                if mask.sum() > 0:
                    key = 'obj{}'.format(i)
                    names[key] = {
                        'id': np.where(mask)[0],
                        'bbox': None,
                    }


        self.names = names
        self.tokens = []
        self.views = views
        self.kwargs = kwargs