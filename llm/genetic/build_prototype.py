# https://github.com/hzaskywalker/Concept/blob/chaoyi2-Apr10/CY/data_collection/build_prototype.ipynb

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import alphashape

import llm.envs.soft_utils as geom 
from llm.envs.soft_utils import cylinder, compute_n_particles
from shapely.geometry import Point

npoints = 2048


def cone(args, center=(0.5, 0.5, 0.5), n=None):
    Rmax, H, ratio = args
    assert 0 <= ratio < 1, 'R_top should be less than Rbottom'
    Rmin = Rmax * ratio

    p = cylinder((Rmax, H), center=(0,H/2,0), n=n)

    y = p[:, 1]
    hidden_top = H * Rmin / (Rmax-Rmin)
    hidden_H = H + hidden_top
    r = (hidden_H - y) / hidden_H * Rmax

    p[:, 0] = p[:, 0] / Rmax * r
    p[:, 2] = p[:, 2] / Rmax * r
    
    p += np.array(center)
    return p

def sample_poly2d_points(poly2d, n):
    minx, miny, maxx, maxy = poly2d.bounds
    xs, ys = [], []
    while len(xs) < n:
        x = np.random.uniform(low=minx, high=maxx, size=1)
        y = np.random.uniform(low=miny, high=maxy, size=1)
        point2d = Point(x, y)
        if poly2d.intersects(point2d):
            xs.append(point2d.x)
            ys.append(point2d.y)
    xs, ys = np.array(xs), np.array(ys)
    return xs, ys
    
def polygon3d(args, center=(0.5, 0.5, 0.5), n=None):
    poly2d, h = args[:2]
    ratio2d = args[2] if len(args) > 2 else 1
    n = n or compute_n_particles(poly2d.area * h)

    ys = np.random.random(n) * h
    xs, zs = sample_poly2d_points(poly2d, n)
    p = np.c_[xs, ys, zs]

    normalize = lambda x: (x - np.min(x)) * 2 / (np.max(x) - np.min(x)) - 1
    p[:, 0] = normalize(p[:, 0]) * ratio2d
    p[:, 2] = normalize(p[:, 2]) * ratio2d
    p[:, 1] = normalize(p[:, 1]) * h

    p = p + np.array(center)
    return p

def save_pcd(xyz, path):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(path, pcd)

def random_sampler(upper, lower=0.03):
    return np.random.random() * (upper - lower) + lower

def plot_points2d(points2d):
    fig, ax = plt.subplots()
    ax.scatter(*zip(*points2d))
    plt.show()



sphere = geom.sphere(1.0, n=npoints)
save_pcd(sphere, 'sphere.pcd')



box = geom.box([1,2,3], n=npoints)
save_pcd(box, 'box.pcd')

cube = geom.box(2, n=npoints)
save_pcd(cube, 'cube.pcd')



r, h = 0.1, 0.5
_cylinder = geom.cylinder((r, h), n=npoints)
save_pcd(_cylinder, 'cylinder.pcd')



Rmax, h, ratio = 0.1, 0.5, 0
_cone = cone((Rmax, h, ratio), n=npoints)
save_pcd(_cone, 'cone.pcd')



Rmax, h, ratio = 0.1, 0.5, 0.5
cone_frustum = cone((Rmax, h, ratio), n=npoints)
save_pcd(cone_frustum, 'cone_frustum.pcd')



triangle = [(0., 0.), (0., 1.), (.5, .5)]
plot_points2d(triangle)
triangle = alphashape.alphashape(triangle, 2)
plt.plot(*triangle.exterior.xy)
triangle = polygon3d((triangle, 0.5), n=npoints)
save_pcd(triangle, 'triangle.pcd')



trapezoid = [(0., 0.), (0., 1.), (1., .25), (1., .75)]
plot_points2d(trapezoid)
trapezoid = alphashape.alphashape(trapezoid, 1)
plt.plot(*trapezoid.exterior.xy)
trapezoid = polygon3d((trapezoid, 0.5), n=npoints)
save_pcd(trapezoid, 'trapezoid.pcd')



L_shape1 = [(0., 0.), (0., 1./3), (0., 2./3), (0., 1.), 
(1./3, 1.), 
(1./3, 2./3), 
(1./3, 1./3), (1./3, 0.)]
L_shape2 = [(0., 2./3), (0., 1.), 
(1./3, 1.), (2./3, 1.), 
(2./3, 2./3), (1./3, 2./3)]
#plot_points2d(L_shape1)
#plot_points2d(L_shape2)
L_shape1 = alphashape.alphashape(L_shape1, 1.)
L_shape2 = alphashape.alphashape(L_shape2, 1.)
L_shape = L_shape1.union(L_shape2)

plt.plot(*L_shape.exterior.xy)
L_shape = polygon3d((L_shape, 0.5), n=npoints)
save_pcd(L_shape, 'L_shape.pcd')





T_shape2 = [(0., 1./3), (0., 2./3), (2./3, 1./3),
(2./3, 2./3), (1./3, 2./3), 
(1./3, 1./3), ]
#plot_points2d(T_shape2)

T_shape2 = alphashape.alphashape(T_shape2, 1.)
T_shape = L_shape1.union(T_shape2)
plt.plot(*T_shape.exterior.xy)
T_shape = polygon3d((T_shape, 0.5), n=npoints)
save_pcd(T_shape, 'T_shape.pcd')

