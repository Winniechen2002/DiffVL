import torch 
import numpy as np
from numpy.typing import NDArray
import open3d as o3d


import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

def scalar_to_rgb(value: NDArray[np.float32], min_value: float, max_value: float, colormap:str ='viridis'):
    normalized_value = (value - min_value) / (max_value - min_value)
    cmap = plt.get_cmap(colormap) # type: ignore
    rgb_color = cmap(normalized_value)
    return rgb_color


def tonumpy(x: torch.Tensor | NDArray[np.float64]) -> NDArray[np.float64]:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def totensor(x: torch.Tensor | NDArray[np.float64], device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def save_pcd(xyz: torch.Tensor|o3d.geometry.PointCloud, path, color=None) -> None:
    if not isinstance(xyz, o3d.geometry.PointCloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tonumpy(xyz))
    else:
        pcd = xyz
    pcd: o3d.geometry.PointCloud = pcd

    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(tonumpy(color))
    o3d.io.write_point_cloud(path, pcd)


OT_SOLVER=None
def get_ot_solver():
    global OT_SOLVER
    if OT_SOLVER is None:
        from geomloss import SamplesLoss
        OT_SOLVER = SamplesLoss(loss='sinkhorn', p=1, blur=0.001)
    return OT_SOLVER


norm_vec = lambda x: x / np.linalg.norm(x)


@torch.no_grad()
def match(pos: torch.Tensor, goal: torch.Tensor):
    ot_solver = get_ot_solver()
    ot_solver.potentials = True
    F, G = ot_solver(totensor(pos, 'cuda:0'), totensor(goal, 'cuda:0'))
    ot_solver.potentials = False
    return F[0]

def calculate_transport_gradient(x: torch.Tensor, y: torch.Tensor):
    x = x.clone().requires_grad_()
    ot_solver = get_ot_solver()
    L = ot_solver(x, y)
    [g] = torch.autograd.grad(L, [x])
    return g / torch.norm(g, dim=1, keepdim=True)


def draw_potentials_on_pts(_points: torch.Tensor, _potentials: torch.Tensor, normals=None):
    # from Sizhe's code
    import open3d as o3d
    points = tonumpy(_points)
    potentials = tonumpy(_potentials)
    normals = tonumpy(normals) if normals is not None else None

    def points_to_pcd(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

        
    def rgb_np(minimum, maximum, value):
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value - minimum) / (maximum - minimum)
        b = (255 * (1 - ratio)).clip(min=0)
        r = (255 * (ratio - 1)).clip(min=0)
        g = 255 - b - r
        return np.stack([r, g, b], axis=-1) / 255

    pcd = points_to_pcd(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_np(potentials.min(), potentials.max(), potentials))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd