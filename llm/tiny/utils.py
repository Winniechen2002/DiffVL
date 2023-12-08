# some utils.. 
from envs.soft_utils import rgb2int
import trimesh


COLOR_MAP = {
    'red': rgb2int(255, 0, 0),
    'green': rgb2int(0, 255, 0),
    'blue': rgb2int(0, 0, 255),
    'white': rgb2int(255, 255, 255),
    'grey': rgb2int(50, 50, 50),
}

COLOR_MAP_INV={v:k for k, v in COLOR_MAP.items()}


def capsule2pcd(size):
    r=size[0]; h=size[1]
    mesh = trimesh.creation.capsule(height=h, radius = r)
    v = mesh
    return trimesh.Trimesh(v.vertices[:, [0, 2, 1]], v.faces)