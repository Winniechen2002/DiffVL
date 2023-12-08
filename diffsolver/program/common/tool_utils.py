from .common import register, SceneSpec, TType


@register('tooly', lang='the y axis of the tool') # height of the tool
def tooly():
    def _height(scene: SceneSpec):
        return scene.tool_space.get_xyz(scene.get_obs()['qpos'])[1]
    return _height

from .numerical import gt
@register('tool_above', lang='the tool is above {}', default_weight=100.)
def tool_above(height: float):
    return gt(tooly(), height)