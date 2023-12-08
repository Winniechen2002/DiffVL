from ..pl import *
import torch
from .utils import COLOR_MAP


dsl = DSL()

mybool = dsl.bool
myint = dsl.int

length = dsl.operators['length']


size_type = EnumType(['small', 'large'], 'Size')
color_type = EnumType(list(COLOR_MAP.keys()), "Color")
shape_type = EnumType(['sphere', 'block', 'mat'], "ShapeType")


tA = Type('\'A')

@dsl.as_primitive
def show(a: tA) -> tA:
    print(a)
    return a

@dsl.as_primitive
def embed(a: tA) -> tA:
    from IPython import embed; embed()
    return a

@dsl.as_primitive
def show_grad(a: tA) -> tA:
    from .dtypes.logprob import LogProb
    if isinstance(a, torch.Tensor):
        a.register_hook(print)
    elif isinstance(a, LogProb):
        a.value.register_hook(print)
    else:
        raise NotImplementedError(f"the show gradient function is not defined on {type(a)}")
    return a
