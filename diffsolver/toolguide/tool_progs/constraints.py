import torch
import numpy as np
from typing import Tuple, Optional
from .utils import SceneSpec, libparser, ToolSpace
from ...program.parser import Parser
from ...program.common import SoftBody, TType, get_goal, get_iobj
from ...program.common.pcd import OBJ, softbody2tensor
import warnings

# CONS, the lower the better
_TOOL_CONS_FACTORY = Parser[[SceneSpec], Tuple[torch.Tensor, Optional[torch.Tensor]]](**libparser._LIBRARIES)

register_library = _TOOL_CONS_FACTORY.register
get_tool_cons = _TOOL_CONS_FACTORY.parse
register_constant = _TOOL_CONS_FACTORY.register_constant

    
# for test only
@register_constant('lower_than_015')
def lower_than_015(scene: SceneSpec):
    qpos = scene.tool_space.get_xyz(scene.get_obs()['qpos'])
    return qpos[1], qpos[1] < 0.15


@register_library('ty') # height of the tool
def ty():
    def _height(scene: SceneSpec):
        return scene.tool_space.get_xyz(scene.get_obs()['qpos'])[1]
    return _height


@register_constant('collision_free')
def collision_free(scene: SceneSpec):
    dist = scene.get_obs()['dist'].min()
    return -dist, dist > 0.0

@register_constant('minimize_dist')
def minimize_dist(scene: SceneSpec):
    dist = scene.get_obs()['dist'].min()
    return dist, None

@register_constant('sample_above')
def sample_above(scene: SceneSpec, height: float = 0.02):
    y = scene.tool_space.get_xyz(scene.get_obs()['qpos'])[1]
    return torch.tensor(0), y > height

@register_constant('control_gap')
def control_gap(scene: SceneSpec, gap_min: float = 0.01, gap_max: float = 0.05):
    warnings.warn("control_gap is deprecated, use gap instead", DeprecationWarning)
    gap = scene.tool_space.get_gap(scene.get_obs()['qpos'])
    return torch.tensor(0), gap < gap_max and gap > gap_min


@register_library('touch_pcd')
def touch_pcd(pcd: OBJ):
    def _touch_pcd(scene: SceneSpec):
        from diffsolver.utils.sdf import compute_sdf
        _pcd = softbody2tensor(pcd, scene)
        d = compute_sdf(scene.env, scene.get_obs()['qpos'], _pcd)
        return d.min(dim=-1)[0].sum(), None
    return _touch_pcd


VERBOSE = False

@register_library('cpdeform')
def cpdeform(objname: str|int): #, check_convexity=False):
    from ..cpdeform import match
    _SCORE = None #TODO: cache the weight matrix, very dangerous I think ..
    HULL = None
    TRI = None

    def _cpdeform(scene: SceneSpec):
        nonlocal _SCORE
        nonlocal HULL, TRI

        if _SCORE is None:
            obj = get_iobj(objname)(scene)
            goal = get_goal(objname)(scene)
            _SCORE = match(obj.pcd(), goal.pcd())

            if VERBOSE:
                from ..cpdeform import save_pcd, scalar_to_rgb

                _obj = obj.pcd().detach().cpu().numpy()
                _goal = goal.pcd().detach().cpu().numpy()
                c_obj = scalar_to_rgb(_SCORE.detach().cpu().numpy(), float(_SCORE.min()), float(_SCORE.max()), colormap='jet')[:, :3]
                goal_obj = np.zeros_like(c_obj)
                goal_obj[:, 1] = 0.5
                save_pcd(_goal, 'tmp/goal.pcd')
                save_pcd(_obj, 'tmp/cur.pcd')
                p = np.r_[_obj, _goal]
                p = p - p.mean(0)
                save_pcd(p, 'tmp/obj.pcd', color=np.r_[c_obj, goal_obj])

        #     if check_convexity:
        #         from scipy.spatial import ConvexHull, Delaunay
        #         points = obj.pcd().detach().cpu().numpy()
        #         HULL = ConvexHull(points)
        #         hull_points = points[HULL.vertices]
        #         TRI = Delaunay(hull_points)

        #         # is_inside = tri.find_simplex(test_point) >= 0

        # if check_convexity:
        #     xyz = scene.tool_space.get_xyz(scene.get_obs()['qpos']).detach().cpu().numpy()
        #     assert TRI is not None
        #     if TRI.find_simplex(xyz) < 0:
        #         return 1e10, None
    

        dists = scene.get_obs()['dist']
        min_dists, dists_index = dists.min(dim=1)

        sort_index = min_dists.argsort()
        total = 0.
        K = max(int(len(dists) * 0.05), 1)

        for i in range(dists.shape[1]):
            tool_index = sort_index[dists_index[sort_index] == i]
            if len(tool_index) > 0:
                ind = tool_index[:K]
                d = min_dists[ind]
                # print('ind.shape', ind.shape, 'score shape', _SCORE.shape, _SCORE[ind])
                assert (d[1:] >= d[:-1]).all()
                assert (dists_index[tool_index] == i).all()
                total = max(total, ((d < 0.05).float() * _SCORE[ind]).sum())
        return -total, None
    return _cpdeform