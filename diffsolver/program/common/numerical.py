# numerical value, comparing two values 
import torch
from .common import register, SceneSpec, Callable, Tuple, STEP_COND


SCALAR = Callable[[SceneSpec], torch.Tensor] | float | torch.Tensor


def SCALAR2Tensor(x: SCALAR, scene: SceneSpec):
    if isinstance(x, float) or isinstance(x, torch.Tensor):
        return x
    else:
        return x(scene)


@register(lang='{} is less than {}')
def lt(x: SCALAR, y: SCALAR) -> STEP_COND:
    def _less(scene: SceneSpec):
        left, right = SCALAR2Tensor(x, scene), SCALAR2Tensor(y, scene)
        loss = left - right
        flag = left < right
        assert isinstance(loss, torch.Tensor)
        assert isinstance(flag, torch.Tensor)
        return torch.relu(loss), flag
    return _less


@register(lang='{} is greater than {}')
def gt(x: SCALAR, y: SCALAR) -> STEP_COND:
    def _less(scene: SceneSpec):
        left, right = SCALAR2Tensor(x, scene), SCALAR2Tensor(y, scene)
        loss = right - left
        flag = left > right
        assert isinstance(loss, torch.Tensor)
        assert isinstance(flag, torch.Tensor)
        return torch.relu(loss), flag
    return _less


@register(lang='the sum of {} and {}')
def add(x: SCALAR, y: SCALAR):
    def _add(scene: SceneSpec):
        left, right = SCALAR2Tensor(x, scene), SCALAR2Tensor(y, scene)
        return left + right
    return _add


@register('del', lang='the difference of {} and {}')
def delete(x: SCALAR, y: SCALAR):
    def _del(scene: SceneSpec):
        left, right = SCALAR2Tensor(x, scene), SCALAR2Tensor(y, scene)
        return left - right
    return _del