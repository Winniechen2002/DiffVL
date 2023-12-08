from .common import TypeVar, SceneSpec, Callable, Tuple, register, List


T1 = TypeVar('T1')
T2 = TypeVar('T2')
@register()
def make_tuple(x: Callable[[SceneSpec], T1], y: Callable[[SceneSpec], T2]) -> Callable[[SceneSpec], Tuple[T1, T2]]:
    def _tuple(scene: SceneSpec):
        return x(scene), y(scene)
    return _tuple


_T = TypeVar('_T')
@register()
def concat(*args: Callable[[SceneSpec], List[_T]]) -> Callable[[SceneSpec], List[_T]]:
    def _concat(scene: SceneSpec):
        x = [arg(scene) for arg in args]
        return sum(x, [])
    return _concat