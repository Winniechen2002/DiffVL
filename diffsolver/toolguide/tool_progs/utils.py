from typing import List, Mapping, Any, Callable
from ...program.scenes import SceneTuple
from ...program.types import SceneSpec, SoftBody, ToolSpace
from ...program.common import libparser #, obj_pos


# def obj(index: int):
#     def get_obj(x: SceneSpec):
#         return x.obj(str(index))
#     return get_obj

# def obj_pos(index: int):
#     def _obj_pos(x: SceneSpec):
#         return x.obj(str(index)).pcd().mean(dim=0)
#     return _obj_pos

# def xyz():
#     def _xyz(scene: SceneSpec):
#         return scene.tool_space.get_xyz(scene.get_obs()['qpos'])
#     return _xyz
    
# SpecFN = Callable[[SceneSpec], Any]

    
# COMMON_LIBRARY: Mapping[str, Callable[..., Callable[[SceneSpec], Any]]] = {
#     'xyz': xyz,
#     'obj': obj,
#     'obj_pos': obj_pos,
#     'tuple': make_tuple,
#     'cat': concat,
#     'less': less,
# }