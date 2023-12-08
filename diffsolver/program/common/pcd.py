# handle part
import numpy as np
import torch
from typing import Callable
from .common import register, SceneSpec, TType, get_iobj, SHAPE_TYPE
from .spartial import POS, py, px
from .numerical import gt
from ..types import SoftBody
from pytorch3d.transforms import matrix_to_euler_angles

from ..constant import *
from enum import Enum


OBJ = int| str | SHAPE_TYPE | TType[torch.Tensor] | torch.Tensor # object


def get_softbody(x: OBJ, scene: SceneSpec):
    out = None
    softbody = None
    if isinstance(x, int) or isinstance(x, str):
        out = get_iobj(x)(scene).pcd()
    elif isinstance(x, torch.Tensor):
        out = x
    else:
        obj = x(scene)
        if isinstance(obj, SoftBody):
            out = obj.pcd()
            softbody = obj
        else:
            out = obj
    return out, softbody


def softbody2tensor(x: OBJ, scene: SceneSpec):
    out = get_softbody(x, scene)[0]
    return out


@register(lang='the center of mass of {}')
def com(x: OBJ):
    def _com(scene: SceneSpec):
        return softbody2tensor(x, scene).mean(dim=0)
    return _com

@register('pcd', lang='the pcd of {}')
def pcd_(x: OBJ):
    def _pcd(scene: SceneSpec):
        return softbody2tensor(x, scene)
    return _pcd


def make_part_fn(name: str, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], default_ratio: float|None=None):
    @register(name, lang=f'the {name} of' + '{}')
    def _part(x: OBJ, ratio: float|None=default_ratio):
        def _fn(scene: SceneSpec):
            pcd, softbody = get_softbody(x, scene)

            if ratio is None:
                com = pcd.mean(dim=0)
            else:
                lower = pcd.min(dim=0)[0]
                upper = pcd.max(dim=0)[0]
                com = lower  * (1-ratio) + upper * ratio
                
            ind = fn(pcd, com)
            if softbody is not None:
                return SoftBody(pcd[ind], softbody.indices[ind])
            return pcd[ind]
        return _fn
    return _part


make_part_fn('leftpart', (lambda pcd, com: pcd[:, 0] < com[0]))
make_part_fn('rightpart', (lambda pcd, com: pcd[:, 0] >= com[0]))
make_part_fn('backpart', (lambda pcd, com: pcd[:, 2] > com[2]))
make_part_fn('frontpart', (lambda pcd, com: pcd[:, 2] < com[2]))
make_part_fn('uppart', (lambda pcd, com: pcd[:, 1] > com[1]))

make_part_fn('leftend', (lambda pcd, com: pcd[:, 0] < com[0]), default_ratio=0.1)
make_part_fn('rightend', (lambda pcd, com: pcd[:, 0] >= com[0]), default_ratio=0.9)
make_part_fn('backend', (lambda pcd, com: pcd[:, 2] > com[2]), default_ratio=0.9)
make_part_fn('frontend', (lambda pcd, com: pcd[:, 2] < com[2]), default_ratio=0.1)
make_part_fn('uppersurface', (lambda pcd, com: pcd[:, 1] > com[1]), default_ratio=0.9)



@register('above', lang="the {} is above {}")
def above(pos: OBJ, height: float):
    return gt(py(com(pos)), height)

# @register('isright', lang="the {} is on the right of {}")
# def isright(pos: OBJ, value: float):
#     return gt(px(com(pos)), value)

# from ..operators.utils import chamfer, emd
# @register('emd', lang='earth mover distance between {} and {}', default_weight = emd_weight)
# def emd_(shape: OBJ, target: OBJ, eps: float=0.001):
#     def _shape_match(x: SceneSpec):
#         dist = emd(softbody2tensor(shape, x), softbody2tensor(target, x).detach())
#         return dist, dist < eps
#     return _shape_match



@register('fix_shape', lang = "keep the shape of the object {}", default_weight = fix_shape_weight)
def fix_shape(body: TType[SoftBody], eps: float=0):
    def _fix_shape(scene: SceneSpec):
        #pcd = softbody2tensor(pos, scene)
        obj = body(scene)
        cur = obj.pcd()
        init = scene.obs[0]['pos'][obj.indices]

        cur = cur - cur.mean(dim=0, keepdim=True)
        init = init - init.mean(dim=0, keepdim=True)

        rigidity = (((cur - init) **2).sum(dim=-1)).mean()
        return rigidity, rigidity < eps

    return _fix_shape

@register('fix_place', lang = "keep the place of the object {}", default_weight = fix_place_weight)
def fix_place(body: TType[SoftBody], eps: float=0):
    def _fix_place(scene: SceneSpec):
        #pcd = softbody2tensor(pos, scene)
        obj = body(scene)
        cur = obj.pcd()
        init = scene.obs[0]['pos'][obj.indices]

        rigidity = (((cur - init) **2).sum(dim=-1)).mean()
        return rigidity, rigidity < eps

    return _fix_place

@register('touch', default_weight = touch_weight)
def touch(pcd: OBJ, eps: float=0.01):
    from .common import touch_v0, get
    if isinstance(pcd, int) or isinstance(pcd, str):
        return touch_v0(get(pcd), eps)

    def _touch_pcd(scene: SceneSpec):
        from diffsolver.utils.sdf import compute_sdf
        #_pcd = softbody2tensor(pcd, scene)
        assert not isinstance(pcd, int) and not isinstance(pcd, str)

        if not isinstance(pcd, torch.Tensor):
            _pcd = pcd(scene)
        else:
            _pcd = pcd
        if isinstance(_pcd, SoftBody):
            obs = scene.get_obs()
            dist = obs['dist'][_pcd.indices]
        else:
            dist = compute_sdf(scene.env, scene.get_obs()['qpos'], _pcd)

        dist: torch.Tensor = torch.relu(dist.min(dim=0)[0] - eps)
        dist = dist.sum(dim=0)
        return dist, dist < eps
    return _touch_pcd





def _relative_pose(a: torch.Tensor, b: torch.Tensor):
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    P = a[None, :]
    Q = b[None, :]

    P1=P-P.mean(dim=1,keepdim=True)
    Q1=Q-Q.mean(dim=1, keepdim=True)
    C = torch.bmm(P1.transpose(1,2), Q1)## B*3*3

    U, S, V = torch.svd(C)
    U = U.transpose(1, 2)
    d = (torch.det(V) * torch.det(U)) < 0.0
    d = - d.float().detach() * 2 + 1 # d == 1 -> -1; d == 0 -> 1
    V = torch.cat((V[:,:-1],  d[:, None, None] * V[:,-1:]), dim=1)
    R = torch.bmm(V, U)

    # T=Q-torch.bmm(R,P.transpose(1,2)).transpose(2,1)
    cors= torch.bmm(R, P.transpose(1, 2)).transpose(2, 1)
    T = (Q - cors).mean(dim=-2)
    return torch.cat((R, T[:,:,None]), dim=2)[0] # return a 3x4 matrix

# def _pose2rpy(relative_pose, axis) -> torch.Tensor:
#     from pytorch3d.transforms import matrix_to_euler_angles
#     axis = torch.tensor(np.array(axis), device=relative_pose.device)
#     return matrix_to_euler_angles(relative_pose[:3,:3], 'XYZ')[axis]

# def _pose_dist(relative_pose, rot_axis_angle) -> torch.Tensor:
#     from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
#     rot_matrix = axis_angle_to_matrix(rot_axis_angle)
#     return torch.arccos((torch.trace(torch.linalg.inv(relative_pose[:3,:3]) @ rot_matrix) - 1)/2)


@register('pose_rpy')
def pose_rpy(x: OBJ, y: OBJ):
    def _rpy(scene: SceneSpec):
        A = softbody2tensor(x, scene)
        B = softbody2tensor(y, scene)
        matrix = _relative_pose(A, B)
        return matrix_to_euler_angles(matrix[:3,:3], 'XYZ')
    return _rpy


def near_pairs(pcd):
    with torch.no_grad():
        from pytorch3d.ops import ball_query
        R = 0.02 #12
        delta = torch.tensor(np.array([
            [-1., 0., 0.],
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 1., 0.],
            [0., 0., -1.],
            [0., 0., 1.],
        ]))
        delta = delta / torch.linalg.norm(delta, axis=-1)[:, None]
        queries = pcd[:, None]
        _, idx, _ = ball_query(queries.reshape(
            1, -1, 3), pcd[None, :], K=10, radius=R+0.009, return_nn=True)
        ind = idx.squeeze(0).reshape(-1) # print(ind)
        ind = torch.vstack([torch.tensor([_ for _ in range(len(pcd)) for i in range(10)], device='cuda:0'), ind])
    return ind

    
    
@register('no_break', lang='preserving the object {} to prevent it from breaking', default_weight = no_break_weight)
def no_break(x: OBJ, eps: float=0.01):
    _index = None

    def _no_break(scene: SceneSpec):
        #if _index is None:
        pcd = softbody2tensor(x, scene)

        nonlocal _index
        if scene.cur_step == 0:
            _index = near_pairs(pcd)

        ind = _index
        assert ind is not None
        mx, _ = torch.linalg.norm(
            pcd[ind[0].long()] - pcd[ind[1].long()], axis=-1
        ).max(axis=-1)
        assert isinstance(mx, torch.Tensor)
        return torch.relu(mx - eps), mx < eps
    return _no_break