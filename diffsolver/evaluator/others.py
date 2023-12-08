from typing import Dict
from torch import Tensor
from diffsolver.evaluator.eval_cfg import PluginConfig
from diffsolver.program.types import OBSDict, SceneSpec
from .evaluator import PluginBase
from .eval_cfg import SaveSceneConfig, CodelineConfig


class SaveScenePlugin(PluginBase):
    def __init__(self, scene: SceneSpec, config: SaveSceneConfig) -> None:
        super().__init__(scene, config)
        self.config = config

        self.init_image = None
        self.goal_image = None
        self.dumped = False


    def reset(self, obs: OBSDict) -> None:
        if self.init_image is None:
            self.init_image = self.scene.env.render('rgb_array')
            self.goal_image = self.scene.state_tuple.get_goal_image()

    def onstep(self, obs: OBSDict, action: Tensor, __locals) -> None:
        pass

    def onfinish(self):
        if not self.dumped:
            self.dumped = True
            return {
                '_images':{
                    'init': self.init_image,
                    'goal': self.goal_image,
                }
            }
        else:
            return {}

            
class CodePlugin(PluginBase):
    # THIS can only work with RL
    def __init__(self, scene: SceneSpec, config: CodelineConfig) -> None:
        super().__init__(scene, config)
        self.config = config
        self.infos = {}

    def reset(self, obs: OBSDict) -> None:
        pass

    def onstep(self, obs: OBSDict, action: Tensor, __locals) -> Dict | None:
        c = __locals['constraints']

        for c in c.tolist():
            if c['code'] not in self.infos:
                self.infos[c['code']] = 0
            else:
                self.infos[c['code']] += float(c['loss'])

    def onfinish(self) -> Dict:
        info = self.infos
        self.infos = {}
        return info
