import torch
import numpy as np
from numpy.typing import NDArray
from diffsolver.utils import MultiToolEnv
from pytorch3d.transforms import quaternion_apply, quaternion_invert

"""
if(type == 0){
    // box
    vec3 q = abs(gx) - vec3(shape_args.w, shape_args.x, shape_args.y);
    sdf = length(max(q, 0.f)) + fminf(fmaxf(fmaxf(q.x, q.y), q.z), 0.f);
}
else if(type == 1){
    vec3 p2 = gx;
    float r=shape_args.w, h=shape_args.x;
    p2.y += h / 2;
    p2.y -= fminf(fmaxf(p2.y, 0.f), h) ;
    sdf = length(p2) - r;
}
"""

def _compute_sdf(tp: int, size: NDArray[np.float32], pos: torch.Tensor, rot: torch.Tensor, points: torch.Tensor):
    # compute t
    invq = quaternion_invert(rot)
    gx = quaternion_apply(invq, points - pos)

    shape_args = torch.tensor(size, dtype=torch.float32)
    if tp == 0:
        q = torch.abs(gx) - shape_args[0:3]
        sdf = torch.linalg.norm(torch.relu(q), dim=-1) + q.max(dim=-1).values.clamp(-np.inf, 0)
    elif tp == 1:
        p2 = torch.clone(gx)
        r, h = shape_args[0], shape_args[1]
        p2[..., 1] += h / 2
        p2[..., 1] -= torch.clamp(p2[..., 1], 0., float(h))
        sdf = torch.linalg.norm(p2, dim=-1) - r
    else:
        raise NotImplementedError
    return sdf 

def compute_sdf(env: MultiToolEnv, qpos: torch.Tensor, points: torch.Tensor):
    assert env.tool_cur is not None
    pos, rot = env.tool_cur.forward_kinematics(qpos)
    tp = {'Box': 0, 'Capsule': 1}[env.tool_cur._cfg.mode] # type: ignore
    size = env.tool_cur._cfg.size # type: ignore

    outs = []
    for p, r in zip(pos, rot):
        outs.append(_compute_sdf(tp, size, p, r, points))
    return torch.stack(outs, dim=1)