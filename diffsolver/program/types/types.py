import torch
from typing import Callable, Tuple, TypedDict, TypeVar
from .softbody import SoftBody
from .scene import SceneSpec
from .tool import Tool
from .constraints import Constraint


SHAPE_TYPE = Callable[[SceneSpec], SoftBody]
TOOL_TYPE = Callable[[SceneSpec], Tool]
STEP_COND = Callable[[SceneSpec], Tuple[torch.Tensor, torch.Tensor]] # one step trajectory -> loss, is_satisfied
SCENE_COND = Callable[[SceneSpec], Constraint] # loss, is_satisfied

POSITION = torch.Tensor

T = TypeVar('T')
TType=Callable[[SceneSpec], T]