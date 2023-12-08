from typing import Dict
from torch import Tensor
from diffsolver.evaluator.eval_cfg import PluginConfig
from diffsolver.program.types import OBSDict, SceneSpec
from .evaluator import PluginBase
from .eval_cfg import IOUConfig


class IOU(PluginBase):
    def __init__(self, scene: SceneSpec, config: IOUConfig, final_goal=None) -> None:
        super().__init__(scene, config)
        self.config = config
        self.env = scene.env
        self.active = scene.goal_obs is not None or final_goal is not None
        if self.active:
            goal = final_goal or scene.goal_obs
            self.goal_grid = self.get_grid(goal['pos'])
            self.target_iou = float(self.compute_soft_iou(goal))

    def get_grid(self, pos: Tensor) -> Tensor:
        assert self.env.get_cur_steps() < self.env.simulator.max_steps - 1, "Cannot compute grid using the last grid"
        grid = self.env.simulator.compute_grid_mass(pos, device='cuda:0')
        assert isinstance(grid, Tensor)
        return grid

    def compute_soft_iou(self, obs):
        goal = (self.goal_grid > 1e-8).float()
        grid = self.get_grid(obs['pos'])
        grid = (grid > 1e-5).float()

        I = (grid * goal).sum()
        return float(I) / (grid.sum() + goal.sum() - I)

    def reset(self, obs: OBSDict) -> None:
        if self.active:
            self.init_iou = float(self.compute_soft_iou(obs))

    def onstep(self, obs: OBSDict, action: Tensor, __locals) -> None:
        if self.active:
            self.last_obs = obs

    def onfinish(self) -> Dict:
        if self.active:
            now = self.compute_soft_iou(self.last_obs)
            return {
                'increase': max(float(now)- self.init_iou, 0.) / self.target_iou,
                'final': float(now),
            }
        else:
            return {}