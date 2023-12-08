import numpy as np
import inspect
from tools import CN


def mu_lam_by_E_nu(E, nu):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lam


def rgb2int(r, g, b):
    return (((r << 8) + g) << 8) + b


def empty_soft_body_state(
    n=None, init=None, color=rgb2int(255, 255, 255)
):
    assert n is not None or init is not None

    if init is None:
        init = np.zeros((n, 3))
    else:
        n = len(init)

    init_v = np.zeros((n, 3))
    F = np.eye(3)[None, :].repeat(n, axis=0)
    C = np.zeros((n, 3, 3))

    particles = [init, init_v, F, C]
    return {
        'particles': particles,
        'softness': 666.,
        'color': np.zeros(n, dtype=np.int32) + color,
        'ids': np.zeros(n, dtype=np.int32)
    }


DEV = 'cuda:0'
N_VOLUME = 10000

def compute_n_particles(volume):
    return int(np.ceil(volume / (0.2**3) * N_VOLUME))

def compute_n_volume(particles):
    return (particles/N_VOLUME)*(0.2**3)

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

def randp(lower, upper=None, method='uniform'):
    return CN(dict(
        lower=lower,
        upper=upper,
        method=method
    ))

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
        outs = []
        for i in childs:
            out, start_index = sample_objects(i, start_index)
            outs.append(out)
        return {i:np.concatenate([k[i] for k in outs]) for i in outs[0]}, start_index
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

        properties = np.zeros((len(out), 3)) - 1
        E = cfg.get("E", -1.)
        nu = cfg.get("nu", -1.)
        yield_stress = cfg.get("yield_stress", -1.)
        properties[:] = np.array([E, nu, yield_stress])

        ids = np.zeros(len(out)) + object_id
        #return out, ids, start_index, np.zeros(len(out)) + color
        return {
            'pos': out,
            'ids': ids,
            'colors': np.zeros(len(out)) + color,
            'properties': properties
        }, start_index

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

    
    
def softbody_sdf(points, rgb):
    from mpm.simulator import MPMSimulator
    from mpm.renderer import Renderer
    simulator = MPMSimulator(1, n_particles=len(points) + 1, max_steps=10)
    simulator.states[0].x.upload(points)

    renderer = Renderer()
    renderer.lookat()

    rgb = np.int32(np.array(rgb) * 255)

    simulator.set_color((rgb[0] * 255 + rgb[1]) * 255 + rgb[2])

    img = renderer.render(simulator, simulator.states[0])

def renderer_mesh(renderer):
    sdf = renderer.sdf_volume.download().reshape(*renderer.voxel_res) - renderer._cfg.sdf_threshold
    color_volume = renderer.color_volume.download().reshape(*renderer.voxel_res, 3)
    bmax, bmin = renderer.bbox_min.download(), renderer.bbox_max.download()

    import mcubes
    vertices, triangles = mcubes.marching_cubes(sdf, 0)
    vertices/=sdf.shape[-1]
    vertices = vertices * (bmax - bmin) + bmin
    mcubes.export_mesh(vertices, triangles, "tmp.dae", "Tmp")