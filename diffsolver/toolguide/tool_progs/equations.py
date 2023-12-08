import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Mapping, Union, Callable, Optional, Any, TypeVar, ParamSpec, ParamSpecArgs
from .utils import List, SceneSpec, libparser
from ...program.parser import Parser
from ...program.types import SoftBody, ToolSpace
from ...program.common import get_iobj, OBJ, com as obj_pos
from ...program.common.pcd import softbody2tensor, OBJ


_TOOL_LIBRARY = Parser[[SceneSpec], Tuple[List[int], List[float|Tuple[float, float]]]](**libparser._LIBRARIES)
register_library = _TOOL_LIBRARY.register
get_tool_eq_fn = _TOOL_LIBRARY.parse

#y = register_library('grasp')

def extract_index(x, index):
    return [x[i] for i in index]


def merge(x, y):
    def _merged(scene: SceneSpec):
        x1, y1 = x(scene)
        x2, y2 = y(scene)
        return x1 + x2, y1 + y2
    return _merged

@register_library('locate')
def locate(pos: OBJ, _width: float| Tuple[float, float, float] = (0.1, 0.1, 0.1)):
    def _locate(scene: SceneSpec):
        tool = scene.tool_space
        x, y, z = tool.get_xyz_index()

        pcd = softbody2tensor(pos, scene)

        target = pcd.mean(dim=0)
        assert target.shape == (3,), "Target position must be 3D."
        width = np.array(_width if isinstance(_width, tuple) else (_width, _width, _width))
        #width = np.minimum(width, (pcd.max(dim=0)[0] - pcd.min(dim=0)[0]).detach().cpu().numpy() * 3)

        return [x, y, z], [(max(float(i)-(width[idx]/2) * int(i!=1), 0.03), float(i) + (width[idx]/2) * (1+ int(i==1))) for idx, i in enumerate(target)]
    return _locate


@register_library('set_coord')
def set_coord(pos: OBJ, axis='xyz'):
    def _set_coord(scene: SceneSpec):
        tool = scene.tool_space
        x, y, z = tool.get_xyz_index()

        pcd = softbody2tensor(pos, scene)

        index = [i for i, a in enumerate('xyz') if a in axis]

        xx = [x, y, z]
        tt = list(map(float, pcd.mean(dim=0)))
        return [xx[i] for i in index], [tt[i] for i in index]
    return _set_coord

@register_library('set_rot')
def set_rot(x, y, z):
    def _set_coord(scene: SceneSpec):
        tool = scene.tool_space
        xx = tool.get_rot_index()
        tt = [x, y, z]
        index = [i for i in range(3) if tt[i] !='?'] 
        return [xx[i] for i in index], [tt[i] for i in index]
    return _set_coord


def convert(min, max=None):
    if max is None:
        return float(min)
    else:
        return (float(min), float(max))

@register_library('roll')
def roll(val: float, max=None):
    def _roll(scene: SceneSpec):
        tool = scene.tool_space
        x = tool.rotx()
        return [x], [convert(val, max)]
    return _roll


@register_library('pitch')
def pitch(val: float, max=None):
    def _pitch(scene: SceneSpec):
        tool = scene.tool_space
        y = tool.roty()
        return [y], [convert(val, max)]
    return _pitch

@register_library('yaw')
def yaw(val: float, max=None):
    def _yaw(scene: SceneSpec):
        tool = scene.tool_space
        z = tool.rotz()
        return [z], [convert(val, max)]
    return _yaw


@register_library('vertical')
def vertical():
    return merge(roll(0.), yaw(0.))

# @register_library('horizontal')
# def horizontal():
#     return roll(np.pi/2)


@register_library('set_gap')
def set_gap(gap_min: float = 0.01, gap_max: float = 0.05):
    def _set_gap(scene: SceneSpec):
        gap = scene.tool_space.get_gap_index()
        return gap, [convert(gap_min, gap_max)]
    return _set_gap


@register_library('isabove')
def isabove(obj_id: int | str):
    def _isabove(scene: SceneSpec):
        y = scene.tool_space.get_xyz_index()[1]
        height = obj_pos(get_iobj(obj_id))(scene)
        return [y], [convert(float(height[1]), 10.)]
    return _isabove 

@register_library('isbehind')
def isbehind(obj_id: int | str):
    def _isbehind(scene: SceneSpec):
        z = scene.tool_space.get_xyz_index()[2]
        com = obj_pos(get_iobj(obj_id))(scene)
        return [z], [convert(float(com[2]), 10.)]
    return _isbehind 

@register_library('isfront')
def isfront(obj_id: int | str):
    def _isfront(scene: SceneSpec):
        z = scene.tool_space.get_xyz_index()[2]
        com = obj_pos(get_iobj(obj_id))(scene)
        return [z], [convert(-10., float(com[2]))]
    return _isfront 


@register_library('isleft')
def isleft(obj_id: int | str):
    def _isleft(scene: SceneSpec):
        x = scene.tool_space.get_xyz_index()[0]
        com = obj_pos(get_iobj(obj_id))(scene)
        return [0], [convert(float(com[0]), 10.)]
    return _isleft 

@register_library('isright')
def isright(obj_id: int | str):
    def _isright(scene: SceneSpec):
        x = scene.tool_space.get_xyz_index()[0]
        com = obj_pos(get_iobj(obj_id))(scene)
        return [0], [convert(-10., float(com[0]))]
    return _isright 

# -------------------


@register_library('grasp')
def grasp(obj_id: int | str, index: Tuple[int, int]=(0, 2)):
    """
    rotx: rotation around x axis
    rotz: rotation around z axis
    roty: rotation around y axis
    """
    def _grasp(scene: SceneSpec):
        tool = scene.tool_space

        pos = obj_pos(get_iobj(obj_id))(scene)
        x, z = extract_index(tool.get_xyz_index(), index)
        x_rot, z_rot = extract_index(tool.get_rot_index(), index)
        x_t, z_t = extract_index(pos, index)

        return [x, z, x_rot, z_rot], [float(x_t), float(z_t), 0., 0.]
    return _grasp

    
@register_library('vgrasp')
def vgrasp(obj: OBJ):
    def _grasp(scene: SceneSpec):
        tool = scene.tool_space

        pos = obj_pos(obj)(scene)

        x, y, z = tool.get_xyz_index()
        x_rot, y_rot, z_rot = tool.get_rot_index()
        x_t, y_t, z_t = pos

        return [x, y, z, x_rot, z_rot, y_rot], [float(x_t), float(y_t), float(z_t), np.pi/2, 0., np.pi/2]
    return _grasp


@register_library('no_rotation')
def no_rotation():
    def _no_rotation(scene: SceneSpec):
        tool = scene.tool_space

        x_rot, y_rot, z_rot = tool.get_rot_index()

        return [x_rot, z_rot, y_rot], [0., 0., 0.]
    return _no_rotation

    
@register_library('xzgrasp')
def xzgrasp(obj: OBJ, index: Tuple[int, int]=(0, 2)):
    """
    rotx: rotation around x axis
    rotz: rotation around z axis
    roty: rotation around y axis
    """
    def _grasp(scene: SceneSpec):
        tool = scene.tool_space

        pos = obj_pos(obj)(scene)
        x, z = extract_index(tool.get_xyz_index(), index)
        x_rot, z_rot = extract_index(tool.get_rot_index(), index)
        x_t, z_t = extract_index(pos, index)

        return [x, z, x_rot, z_rot], [float(x_t), float(z_t), 0., 0.]
    return _grasp

@register_library('xyzgrasp')
def xyzgrasp(obj: OBJ):
    def _grasp(scene: SceneSpec):
        tool = scene.tool_space

        pos = obj_pos(obj)(scene)

        x, y, z = tool.get_xyz_index()
        x_rot, y_rot, z_rot = tool.get_rot_index()
        x_t, y_t, z_t = pos

        return [x, y, z, x_rot, z_rot, y_rot], [float(x_t), float(y_t), float(z_t), 0., 0., 0.]
    return _grasp


@register_library('cgrasp')
def cgrasp(obj: OBJ):
    def _grasp(scene: SceneSpec):
        tool = scene.tool_space

        pos = obj_pos(obj)(scene)

        x, y, z = tool.get_xyz_index()
        x_rot, y_rot, z_rot = tool.get_rot_index()
        x_t, y_t, z_t = pos

        return [x, y, z, x_rot, z_rot, y_rot], [float(x_t), float(y_t), float(z_t), 0., np.pi/2, 0.]
    return _grasp