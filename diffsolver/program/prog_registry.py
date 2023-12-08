from .types.types import Constraint, SceneSpec, SCENE_COND
from typing import Callable, TypedDict, List
from .parser import Parser
from ..config import DiffPhysStageConfig
from .common import *


class ProgType(TypedDict):
    horizon: int
    prog: Callable[[SceneSpec], Constraint]


_PROG_LIBS = Parser[[SceneSpec], Constraint](**libparser._LIBRARIES)
register_prog = _PROG_LIBS.register



def build_prog(prog: DiffPhysStageConfig, stages: List[DiffPhysStageConfig]) -> ProgType:
    if len(stages) == 0:
        fn = _PROG_LIBS.parse(prog.code)
        return ProgType(horizon=prog.horizon, prog=fn)
    else:
        _stages = [build_prog(s, []) for s in stages]
        from .temporal_ops import multi_stages
        progs = [s['prog'] for s in _stages]
        horizons = [s['horizon'] for s in _stages]
        return ProgType(horizon=sum(horizons), prog=multi_stages(progs, horizons))