# the goal of this function is to collect functions that help us build geometric scenes .. 
# https://github.com/hzaskywalker/plb_lang/blob/refactor/plb/lang/envs/tiny3d/csg.py
import inspect
import tqdm
import torch
import numpy as np
import transforms3d
from tools.config import CN, merge_inputs
from itertools import product

DEV = 'cuda:0'
N_VOLUME = 5000

def rgb2int(r, g, b):
    return (((r << 8) + g) << 8) + b


def compute_n_particles(volume):
    return int(np.ceil(volume / (0.2**3) * N_VOLUME))

def vec(args, device=None, dtype=torch.float32):
    device = device or DEV
    if not isinstance(args, torch.Tensor):
        args = torch.tensor(args, device=device, dtype=dtype)
    return args.to(device)

def to_numpy(v):
    if isinstance(v, torch.Tensor):
        v = v.cpu().detach().numpy()
    return np.array(v)

def sphere(radius, center=(0.5, 0.5, 0.5), n=None):
    n = n or compute_n_particles((radius ** 3) * 4 * np.pi / 3)

    p = np.random.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    u = np.random.random(size=(n, 1)) ** (1. / 3)
    p = p * u * radius + np.array(center)
    return p


def box(width, center=(0.5, 0.5, 0.5), n=None):
    try:
        p = float(width)
        width = [width] * 3
    except:
        pass

    width = np.array(width)
    n = n or compute_n_particles(np.prod(width))
    p = (np.random.random((n, 3)) * 2 - 1) * (0.5 * width) + np.array(center)
    return p

def cylinder(args, center=(0.5, 0.5, 0.5), n=None):
    radius, h = args
    n = n or compute_n_particles(radius * radius * np.pi * h)

    p = np.random.normal(size=(n, 2))
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    u = np.random.random(size=(n, 1)) ** (1. / 2)
    p = p * u * radius
    p = np.stack((p[:, 0], (np.random.random(size=(n,)) - 0.5) * h, p[:, 1]), 1)
    p = p + np.array(center)
    return p

def center(p):
    return p.mean(axis=0)

def transport(p, delta):
    return p + vec(delta, device=p)

def rot(p, axis_angle, c=None):
    axis_angle = to_numpy(axis_angle)

    if len(axis_angle) == 3:
        theta = np.linalg.norm(axis_angle)
        if theta > 1e-4:
            axis = axis_angle / theta.clip(1e-10, np.inf)
        else:
            theta = 0
            axis = np.array([1., 0., 0.])

    else:
        theta, axis = axis_angle[0], axis_angle[1:4]

    mat = vec(
        transforms3d.axangles.axangle2mat(
            axis, theta
        ), p.device
    )

    if c is None:
        c = p.mean(axis=0)
    return ((p - c) @ mat.T) + c

def mse(x, y): # not squared
    assert x.shape == y.shape
    return ((x - y) ** 2).mean()

def mean_l1(x, y):
    return torch.abs(x, y).mean()

def mean_l2_dist(x, y):
    return (((x - y) ** 2).sum(axis=0).clamp(1e-12, np.inf)).mean()


def stack(p1, p2):
    return torch.cat((p1, p2))


def area(self, p, a, b):
    x = a
    a = p[:] - x[None, :]
    b = b[None, :] - x[None, :]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return -cross


_emd_fn = None
def emd(a, b, p=1, blur=0.01, *args, **kwargs):
    global _emd_fn
    from geomloss import SamplesLoss
    _emd_fn = _emd_fn or SamplesLoss(loss='sinkhorn', p=2., blur=0.01)
    _emd_fn.p = p
    _emd_fn.blur = blur
    return _emd_fn(a.clone(), b.clone(), *args, **kwargs)

_chamfer = None
def chamfer(a, b, *args, **kwargs):
    global _chamfer
    from chamferdist import ChamferDistance
    _chamfer = _chamfer or ChamferDistance()
    return _chamfer(a[None,:].clone(), b[None,:].clone(), *args, **kwargs)


def shape_dist(a, b, method='chamfer', centered=True):
    shape_dist_fn = chamfer if method == 'chamfer' else emd
    if centered:
        ca = center(a)
        cb = center(b)
        a = a - ca
        b = b - cb
    return shape_dist_fn(a, b)

def center_dist(a, b):
    return mse(center(a), center(b))


def cut(p, a, b):
    x = vec(a, p.device)
    a = p[:] - x[None, :]
    b = b[None, :] - x[None, :]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return cross

def compute_bbox(p: np.ndarray):
    if isinstance(p, torch.Tensor):
        return p.min(axis=0)[0], p.max(axis=0)[0]
    else:
        return p.min(axis=0), p.max(axis=0)


def press(p, left=True, right=True):
    bbox = compute_bbox(p)

    center = (bbox[1][0] + bbox[0][0]) / 2
    height = bbox[1][1]

    if left:
        x = float(bbox[0][0] - 0.05)
        A = vec([x, 0.])
        B = vec([float(center), float(height)])
        p  = p[cut(p[:, [0, 1]], A, B) >= 0]


    if right:
        x = float(bbox[1][0] + 0.05)
        A = vec([x, 0.])
        B = vec([float(center), float(height)])
        p  = p[cut(p[:, [0, 1]], A, B) <= 0]

    return p


def p2g(x, size=64, p_mass=1.):
    if x.dim() == 2:
        x = x[None, :]
    batch = x.shape[0]
    grid_m = torch.zeros(batch, size * size * size, dtype=x.dtype, device=x.device)
    inv_dx = size

    fx = x * inv_dx
    base = (x * inv_dx - 0.5).long()
    fx = fx - base.float()
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[i][..., 0] * w[j][..., 1] * w[k][..., 2] * p_mass
                target = (base + torch.tensor(np.array([i, j, k]), dtype=torch.long, device='cuda:0')).clamp(0, size-1)
                idx = (target[..., 0] * size + target[..., 1]) * size + target[..., 2]
                grid_m.scatter_add_(1, idx, weight)
    return grid_m.reshape(batch, size, size, size)


def compute_sdf(density, eps=1e-4, inf=1e10):
    if density.dim() == 3:
        density = density[None, :, :]
    dx = 1./density.shape[1]
    with torch.no_grad():
        nearest_points = torch.stack(torch.meshgrid(
            torch.arange(density.shape[1]),
            torch.arange(density.shape[2]),
            torch.arange(density.shape[3]),
        ), axis=-1)[None, :].to(density.device).expand(density.shape[0], -1, -1, -1, -1) * dx
        mesh_points = nearest_points.clone()

        is_object = (density <= eps) * inf
        sdf = is_object.clone()

        for i in tqdm.trange(density.shape[1] * 2): # np.sqrt(1^2+1^2+1^2)
            for x, y, z in product(range(3), range(3), range(3)):
                if x + y + z == 0: continue
                def get_slice(a):
                    if a == 0: return slice(None), slice(None)
                    if a == 1: return slice(0, -1), slice(1, None)
                    return slice(1, None), slice(0, -1)
                f1, t1 = get_slice(x)
                f2, t2 = get_slice(y)
                f3, t3 = get_slice(z)
                fr = (slice(None), f1, f2, f3)
                to = (slice(None), t1, t2, t3)
                dist = ((mesh_points[to] - nearest_points[fr])**2).sum(axis=-1)**0.5
                dist += (sdf[fr] >= inf) * inf
                sdf_to = sdf[to]
                mask = (dist < sdf_to).float()
                sdf[to] = mask * dist + (1-mask) * sdf_to
                mask = mask[..., None]
                nearest_points[to] = (1-mask) * nearest_points[to] + mask * nearest_points[fr]
        return sdf



def randp(lower, upper=None, method='uniform'):
    return CN(dict(
        lower=lower,
        upper=upper,
        method=method
    ))


def random(lower, upper=None, method='uniform'):
    if isinstance(lower, dict):
        lower, upper, method = lower['lower'], lower.get(
            'upper', None), lower.get('method', 'uniform')

    assert method == 'uniform'

    if isinstance(lower, float):
        shape = ()
    else:
        shape = (len(lower),)

    lower = np.array(lower)

    if upper is None:
        upper = lower
    upper = np.array(upper)

    return np.random.random(shape) * (upper - lower) + lower

def sample_objects(
    cfg,
    start_index=0,
    parent_id=None,
):

    cfg = CN(cfg)
    childs = cfg.get('childs', None)

    object_id = cfg.get('id', None)
    if object_id is None:
        object_id = parent_id

    if childs is not None:

        pos, ids, colors = [], [], []
        for i in childs:
            a, b, start_index, color = sample_objects(i, start_index)
            pos.append(a)
            ids.append(b)
            colors.append(color)

        return np.concatenate(pos), np.concatenate(ids), np.concatenate(colors)
    else:
        method = cfg.get('method')
        method = eval(method)

        shape_arg_names = list(inspect.signature(method).parameters.keys())[0]

        n = cfg.get("n", None)
        center = random(
            (cfg.get("center", None) or randp((0.5, 0.1, 0.5))))
        
        if shape_arg_names in cfg:
            args = cfg[shape_arg_names]
        else:
            args = random(
                (cfg.get("shape_args", None) or randp(0.1)))

        color = rgb2int(*(cfg.get("color", None) or (255, 255, 255)))

        # object ids..
        out = method(args, center, n)

        if object_id is None:
            object_id = start_index
            start_index += 1

        ids = np.zeros(len(out)) + object_id
        return out, ids, start_index, np.zeros(len(out)) + color

def save_pcd(xyz, path, colors=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        assert xyz.shape[0] == colors.shape[0], f'{xyz.shape[0]} and {colors.shape[0]} shape mismatch.'
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def save_overlap_pcd(path, pc1_dict, pc2_dict):
    pc1_num = pc1_dict['xyz'].shape[0]
    xyz = np.concatenate([pc1_dict['xyz'], pc2_dict['xyz']], axis=0)
    colors = np.zeros(xyz.shape)
    colors[:pc1_num, :] = pc1_dict['color'] 
    colors[pc1_num:, :] = pc2_dict['color']
    save_pcd(xyz, path, colors=colors)

class SoftObject:
    """[summary]
    We need revise the soft object to allow use extract particles of different ids and modify them
    """
    def __init__(
        self, pos, vel=None, dist=None, env=None, grid=None, eigen=None,
    ):
        self.pos = pos
        self.vel = vel
        self.dist = dist

        self.env = env
        self._grid = grid
        self._eigen = eigen


    @property
    def grid(self):
        if self._grid is None:
            self._grid = self._grid or p2g(
                self.pos[None, :], self.env.simulator.n_grid)[0]
        return self._grid

    def dist_to_tool(
        self,
        tool_id=None,
        soft=False,
        clamp=0.01
    ):
        assert not soft
        tool_id = tool_id or slice(None)
        return self.dist[:, tool_id].min(axis=0)[0].clamp(clamp, np.inf)

    def static(self):
        return (self.vel ** 2).mean()

    @property
    def n(self):
        return len(self.pos)


def default_shape_metrics(**kwargs):
    default_cfg = CN(dict(
        chamfer=1.,
        emd=0.,
        # hausdorff=0.,
        grid=0.,
        center=0.,
    ))
    return merge_inputs(
        default_cfg, **kwargs)



def distance_to(obj, goal, shape_metric):
    # compare the shape between the two ..
    #shape_metric = merge_inputs(self._cfg.shape_metric, **kwargs)
    shape_dists = {}
    centered = shape_metric.center > 0

    if shape_metric.chamfer > 0.:
        shape_dists['chamfer'] = shape_dist(
            obj.pos, goal.pos, method='chamfer', centered=centered)

    if shape_metric.center > 0.:
        shape_dists['center'] = center_dist(obj.pos, goal.pos)

    if shape_metric.emd > 0.:
        raise NotImplementedError

    #if shape_metric.hausdorff > 0.:
    #    raise NotImplementedError

    if shape_metric.grid > 0.:
        assert obj.grid.shape == goal.grid.shape, f"{obj.grid.shape} {goal.grid.shape}"
        shape_dists['grid'] = torch.abs(
            obj.grid/obj.grid.sum() - goal.grid/goal.grid.sum()
        ).sum()

    shape = sum([
        shape_metric[i] * shape_dists[i]
        for i in ['chamfer', 'grid', 'center'] if i in shape_dists
    ])
    return shape, {k: float(v) for k, v in shape_dists.items()}
