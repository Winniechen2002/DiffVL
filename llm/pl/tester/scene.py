from ..dsl import DSL
from ..types import Integer, List, Bool, DataType
# from ..lisp import lmap, concat, lfilter, cons, car
#from ..functool import ckattr, ckrel
import numpy as np
from ...envs import WorldState

scene_dsl = DSL()
myint = scene_dsl.int
mybool = scene_dsl.bool

#TODO: support method of a class ..

world_state = DataType(WorldState)
Array = DataType(np.ndarray)
pos = List(myint)


@scene_dsl.as_primitive
def get_scene() -> world_state:
    return WorldState.get_empty_state(n=1000)


@scene_dsl.as_primitive
def get_x(state: world_state) -> Array:
    return state.X


@scene_dsl.as_primitive
def mean(array: Array) -> pos:
    return np.int64(array.mean(axis=0)).tolist()


@scene_dsl.as_primitive
def all_positive(array: Array) -> mybool:
    return bool(array.mean() > 0)


@scene_dsl.as_func
def execute():
    checker = compose(get_x, all_positive)
    return checker(get_scene())



x = execute.pretty_print()
print(x)
exit(0)
y = scene_dsl.parse(x).pretty_print()
assert x == y
print(scene_dsl)
print(scene_dsl.default_executor.eval(execute))