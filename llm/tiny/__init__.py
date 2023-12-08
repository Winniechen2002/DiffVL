from .mydsl import *

from .dtypes.LTL import bind_LTL
bind_LTL(dsl)

from .dtypes import dshape, logprob, LTL

from .scene import Scene, scene_type
from .softbody import SoftBody, soft_body_type
from .tool import Tool, tool_type, tool_name_type
from .library import *
from llm.pl import Executor
from .dtypes.LTL import TStart, TLast, TAll, TExists, Then
from .libpcd import myfloat


def make_executor():
    executor = Executor(dsl) # only execute the returned variables .. 

    dshape.bind(executor)
    logprob.bind(executor)

    dsl.differentiable_executor = executor
    return executor