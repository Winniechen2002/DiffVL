import torch
from .task import Task, randp, default_shape_metrics, random, SoftObject
from .. import geom


class MoveObject(Task):
    def __init__(
        self,
        cfg=None,
        dx=randp((0.1, 0.1, 0.1)),
        rot=randp((0., 0., 0.)),
        shape_weight=1.,
        contact_weight=1.,
        shape_metric=default_shape_metrics(center=10.),
    ) -> None:
        super(MoveObject, self).__init__()

    @torch.no_grad()
    def set_goal(self):
        cur_pos = self._taichi_state['pos'].detach().clone()

        cur_pos[self.ctrl_mask] = geom.rot(
            cur_pos[self.ctrl_mask],
            random(self._cfg.rot)) + geom.vec(random(self._cfg.dx)
                                              )
        self.goals = self.shape_from_state({"pos": cur_pos})