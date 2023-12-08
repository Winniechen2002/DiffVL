from torch import Tensor
from typing import Dict, Type

from diffsolver.program.types import OBSDict, SceneSpec
from .eval_cfg import EvalConfig, PluginConfig
from ..program.types import SceneSpec, OBSDict

    
class PluginBase:
    def __init__(self, scene: SceneSpec, config: PluginConfig) -> None:
        self.scene = scene
        self.config = config

    def reset(self, obs: OBSDict) -> None:
        raise NotImplementedError

    def onstep(self, obs: OBSDict, action: Tensor, __locals) -> None | Dict:
        raise NotImplementedError

    def onfinish(self) -> Dict:
        raise NotImplementedError


# NOTE: Hacks to evaluate the iou across different stages. This is not standard.
FINAL_GOAL = None
FINAL_GOAL_IOU = None

class Evaluator(PluginBase):
    BUILDER: Dict[str, Type[PluginBase]] = {}
    def __init__(self, scene: SceneSpec, config: EvalConfig) -> None:
        from .IOU import IOU
        from .others import SaveScenePlugin, CodePlugin

        self.BUILDER['iou'] = IOU
        self.BUILDER['save_scene'] = SaveScenePlugin
        self.BUILDER['code'] = CodePlugin

        self.config = config
        self.plugins: Dict[str, PluginBase] = {}
        for k in self.config.metrics:
            self.plugins[k] = self.BUILDER[k](scene, getattr(self.config, k))
        
        global FINAL_GOAL, FINAL_GOAL_IOU
        if FINAL_GOAL is not None:
            if FINAL_GOAL_IOU is None:
                from ..program.scenes.scene_tuple import load_goal
                goal_obs, goal_image, _ = load_goal(scene.env, FINAL_GOAL)
                from tools.utils import logger
                logger.savefig('final_goal.png', goal_image)
                FINAL_GOAL_IOU = IOU(scene, self.config.iou, goal_obs)
            self.plugins['final'] = FINAL_GOAL_IOU

    def reset(self, obs: OBSDict) -> None:
        for _, v in self.plugins.items():
            v.reset(obs)

    def onstep(self, obs: OBSDict, action: Tensor, __locals) -> Dict:
        output = {}
        for k, v in self.plugins.items():
            out = v.onstep(obs, action, __locals)
            if out is not None:
                output[k] = out
        return output

    def onfinish(self) -> Dict:
        return {k: v.onfinish() for k, v in self.plugins.items()}