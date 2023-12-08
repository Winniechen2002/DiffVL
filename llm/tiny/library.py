# common function
import numpy as np
from .mydsl import *
from .scene import Scene
from .softbody import soft_body_type
from .libpcd import *


tA = Type('\'A')
tB = Type('\'B')
attr = Arrow(tA, tB)
myint = dsl.int

@dsl.as_func
def be(attr_func: attr, value: tB):
    return compose(attr_func, rcurry(Eq, value))


@dsl.as_primitive
def scale(a: tA, b: myfloat) -> tA:
    if isinstance(a, bool):
        return a
    from .dtypes.logprob import LogProb
    return LogProb(a.value * b)

@dsl.as_primitive
def annealing(a: tA, b: myfloat, c: myfloat) -> tA:
    if isinstance(a, bool):
        return a
    from .dtypes.logprob import LogProb
    return LogProb(a.value * (c + b * dsl.optim_iter / 200.))

from .libpcd import *
from .precond import *