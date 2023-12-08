import torch
import numpy as np
from .task import Task, randp, default_shape_metrics, random, geom, SoftObject

class PressObject(Task):
    def __init__(
        self,
        cfg=None,
        shape_args=randp(0.2),
        left=True,
        right=True,
        shape_weight=1.,
        contact_weight=10.
    ) -> None:
        super(PressObject, self).__init__()

    @torch.no_grad()
    def set_goal(self):
        cur_pos = self._taichi_state['pos'].detach().clone()
        pressed = geom.press(
            cur_pos,
            self._cfg.left,
            self._cfg.right
        )
        sample_idx = np.random.choice(len(pressed), int(self.ctrl_mask.sum()))
        cur_pos[self.ctrl_mask] = pressed[geom.vec(sample_idx, dtype=torch.long)]
        self.goals = self.shape_from_state({"pos": cur_pos})