# handle direction and orientation
# tools in the tensor level
from .common import register, SceneSpec, TType
from .numerical import lt
import torch

POS = torch.Tensor | TType[torch.Tensor] # position
ANGLE = torch.Tensor # 4d angle

def _totensor(x: POS, scene: SceneSpec):
    if isinstance(x, torch.Tensor):
        return x
    else:
        return x(scene)

@register(lang='the x axis of {}')
def px(x: POS):
    def _px(scene: SceneSpec):
        return _totensor(x, scene)[0]
    return _px


@register(lang='the y axis of {}')
def py(x: POS):
    def _py(scene: SceneSpec):
        return _totensor(x, scene)[1]
    return _py

@register(lang='the z axis of {}')
def pz(x: POS):
    def _pz(scene: SceneSpec):
        return _totensor(x, scene)[2]
    return _pz



@register('max', lang='the max axis of {}')
def max_(x: POS):
    def _max(scene: SceneSpec):
        return _totensor(x, scene).max(dim=0)[0]
    return _max


@register('min', lang='the min of {}')
def min_(x: POS):
    def _max(scene: SceneSpec):
        return _totensor(x, scene).min(dim=0)[0]
    return _max

# @register()
# def onleft(a: POS, b: POS):
#     return less(px(a), px(b))

# @register()
# def onright(a: POS, b: POS):
#     return less(px(b), px(a))

# @register()
# def below(a: POS, b: POS):
#     return less(py(a), py(b))

# @register()
# def ontop(a: POS, b: POS):
#     return less(py(b), py(a))


# @register()
# def behind(a: POS, b: POS):
#     return less(pz(a), pz(b))

# @register()
# def infront(a: POS, b: POS):
#     return less(pz(a), pz(b))