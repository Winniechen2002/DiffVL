# common functions shared by diffphys, equality and constraints in tool samplers.
import torch
from ..parser import Parser
from ..operators.utils import chamfer, emd
from ..operators.code import as_code
import  termcolor
from typing import Union, Any, Optional, Callable, TypeVar, List, Tuple
from ..types import SceneSpec, SoftBody, TType, SHAPE_TYPE, POSITION, SCENE_COND, STEP_COND
from ..constant import *


SpecFN = Callable[[SceneSpec], Any]
libparser = Parser[[SceneSpec], Any]()


def register(_name: Optional[str] = None, lang: Optional[str] = None, default_weight: float|None = None):
    def wrapper(func):
        name = _name or func.__name__
        return libparser.register(name)(as_code(func, lang=lang, default_weight=default_weight))
    return wrapper


@register('dy', 'lift {} by a small amount {}')
def dy(A: TType[SoftBody], eps: float):
    def _dy(x: SceneSpec):
        #TODO: hack here, just add a small offset to the y axis
        soft = A(x)
        pcd = soft.pcd()
        return SoftBody(pcd + torch.tensor([0., eps, 0.], device=pcd.device), torch.arange(len(pcd), device=pcd.device))
    return _dy


@register('get_iobj', lang= termcolor.colored('the soft body {}', 'green'))
def get_iobj(index: Union[int, str], timestep: int|None =None):
    def get_body(x: SceneSpec):
        return x.obj(str(index), timestep)
    return get_body

@register('get', lang= termcolor.colored('the soft body {}', 'green'))
def get(index: Union[int, str], timestep: int|None =None):
    def get_body(x: SceneSpec):
        return x.obj(str(index), timestep)
    return get_body


@register('get_others', lang= "the others soft bodies {}")
def get_others(index: Union[int, str], timestep: int|None =None):
    def get_body(x: SceneSpec):
        return x.otherobj(str(index), timestep)
    return get_body

@register('get_objs', lang="the soft bodies {}")
def get_objs(indexs: list[Union[int, str]], timestep: int|None =None):
    def get_bodys(x: SceneSpec):
        pcds = []
        for i in indexs:
            obj = x.obj(str(i), timestep)
            pcds.append(obj.pcd())
        pcd = torch.cat(pcds, dim=0)
        return SoftBody(pcd, torch.arange(len(pcd), device=pcd.device))
    return get_bodys

@register('get_goal', lang=termcolor.colored('target of {}', 'red'))
def get_goal(index: Union[int, str]):
    def get_goal(x: SceneSpec):
        return x.goal(str(index))
    return get_goal

@register('get_all_objs', lang= termcolor.colored('the all the soft bodys', 'green'))
def get_all_objs(timestep: int|None =None):
    def get_body(x: SceneSpec):
        return x.obj('all', timestep)
    return get_body

@register('get_all_goals', lang= termcolor.colored('the all the target of soft bodys', 'red'))
def get_all_goals():
    def get_goal(x: SceneSpec):
        return x.goal('all')
    return get_goal

@register('get_goals', lang=termcolor.colored('targets of {}', 'red'))
def get_goals(index: Union[int, str]):
    def get_goals(x: SceneSpec):
        pcds = []
        for i in indexs:
            obj = x.goal(str(i))
            pcds.append(obj.pcd())
        pcd = torch.cat(pcds, dim=0)
        return SoftBody(pcd, torch.arange(len(pcd), device=pcd.device))
    return get_goals

@register('get_others_goals', lang= "the goal of the others soft bodies {}")
def get_others_goals(index: Union[int, str]):
    def get_body(x: SceneSpec):
        return x.othergoal(str(index))
    return get_body

@register('obj_pos', lang='center of {}')
def obj_pos(obj: SHAPE_TYPE):
    def _obj_pos(x: SceneSpec):
        out = obj(x).pcd().mean(dim=0)
        return out
    return _obj_pos


@register('emd', lang='earth mover distance between {} and {}', default_weight = emd_weight)
def emd_(shape: SHAPE_TYPE, target: SHAPE_TYPE, eps: float=0.001):
    def _shape_match(x: SceneSpec):
        y = shape(x).pcd()
        dist = emd(y, target(x).pcd())
        return dist, dist < eps
    return _shape_match

@register('shape_match', lang='Chamfer distance between {} and {}', default_weight = big_object_weight)
def shape_match(shape: SHAPE_TYPE, target: SHAPE_TYPE, eps: float=0.001):
    def _shape_match(x: SceneSpec):
        dist = chamfer(shape(x).pcd(), target(x).pcd())
        return dist, dist < eps
    return _shape_match

@register('shape_l2', lang='particle distance between {} and {}', default_weight = big_object_weight)
def shape_l2(shape: SHAPE_TYPE, target: SHAPE_TYPE, eps: float=0.001):
    def _shape_match(x: SceneSpec):
        dist = ((shape(x).pcd() - target(x).pcd()) ** 2).sum(dim=-1).mean()
        return dist, dist < eps
    return _shape_match

@register('pcd_l2', lang='particle distance between {} and {}', default_weight = big_object_weight)
def pcd_l2(shape: SHAPE_TYPE, target: SHAPE_TYPE, eps: float=0.):
    def _shape_match(x: SceneSpec):
        A = shape(x).pcd()
        dist = ((A - target(x).pcd()) ** 2).sum(dim=-1).mean()
        if A.shape[0] < 200:
            dist = dist * 10
        return dist, dist < eps
    return _shape_match

@register('lift_up', lang='lift the shape up', default_weight = 2.)
def lift_up(obj: str, height: float = 0.2, eps: float=0.):
    def _lift_up(x: SceneSpec):
        shape = get_iobj(obj)
        y = shape(x).pcd()
        h = y.mean(dim=0, keepdim=True)[0,1]
        return (height - h).clamp_min(0), height > h
    return _lift_up


@register('emd2goal', lang='earth mover distance between an object its goal', default_weight = emd_weight)
def emd2goal(obj: str):
    return emd_(get(obj), get_goal(obj), 0.001)
    
@register('emd_all', lang='earth mover distance between all of the objects and goals', default_weight = emd_weight)
def emd_all(eps: float):
    return emd_(get_all_objs(), get_all_goals(), eps)

@register('shape_match_all', lang='Chamfer distance between all of the objects and goals', default_weight = big_object_weight)
def shape_match_all(eps: float):
    return shape_match(get_all_objs(), get_all_goals(), eps)

@register('shape_l2_all', lang='particle distance between all of the objects and goals', default_weight = big_object_weight)
def shape_l2_all(eps: float):
    return shape_l2(get_all_objs(), get_all_goals(), eps)

def touch_v0(target: SHAPE_TYPE, eps: float=0.01):
    def _tool_close(x: SceneSpec):
        obs = x.get_obs()
        dist = obs['dist']
        dist: torch.Tensor = torch.relu(dist[target(x).indices].min(dim=0)[0] - eps)
        dist = dist.sum(dim=0)
        return dist, dist < eps
    return _tool_close



@register('away', lang='the manipulator away {}', default_weight = away_weight)
def away(target: SHAPE_TYPE, eps: float=0.001, limit: float = 0.1):
    def _tool_away(x: SceneSpec):
        obs = x.get_obs()
        dist = obs['dist']
        dist: torch.Tensor = torch.relu(dist[target(x).indices].min(dim=0)[0] - eps)
        dist = dist.min(dim=0)[0]
        dist = torch.min(dist, torch.tensor(limit))
        return -dist, dist < eps
    return _tool_away


