import transforms3d
import sapien.core as sapien
import torch
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from sapien.core import Pose
import trimesh
import PIL.Image as im
import numpy as np
import skimage.measure as measure
from mpm.simulator import MPMSimulator
from mpm.renderer import Renderer
from diffsolver.paths import FILE_PATH
import os

ASSETS_PATH = os.path.join(FILE_PATH, '../ui/assets')
assert os.path.exists(ASSETS_PATH)

ray_tracing = True

if ray_tracing:
    sapien.render_config.camera_shader_dir = "rt"
    sapien.render_config.viewer_shader_dir = "rt"
    sapien.render_config.rt_samples_per_pixel = 256  # change to 256 for less noise
    sapien.render_config.rt_use_denoiser = False  # change to True for OptiX denoiser

engine = sapien.Engine()  # Create a physical simulation engine
engine.set_log_level('warning')

renderer = sapien.SapienRenderer()  # Create a renderer for visualization
engine.set_renderer(renderer)


scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
scene.set_timestep(1 / 10000.0)  # Set the simulation frequency


simulator = MPMSimulator(1, n_particles=100000, max_steps=10)
mpm_renderer = Renderer(voxel_res=(256, 256, 256))
# renderer.lookat()

def plb2pose(p: torch.Tensor):
    assert p.shape[-1] == 7
    pos = p[..., :3]
    rot = p[..., 3:]

    pose = pos.new(*pos.shape[:-1], 4, 4)
    pose[..., 3, 3] = 1
    pose[..., :3, 3] = pos[..., [0, 2, 1]]
    x = quaternion_to_matrix(rot)
    x = x[..., [0, 2, 1], :]
    x = x[..., :, [0, 2, 1]]
    pose[..., :3, :3] = x
    return pose
    

def softbody_sdf(points, sdf_threshold=0.414):
    simulator.states[0].x.upload(points)
    img = mpm_renderer.render(simulator, simulator.states[0])
    sdf = mpm_renderer.sdf_volume.download().reshape(*mpm_renderer.voxel_res) - sdf_threshold
    #color_volume = renderer.color_volume.download().reshape(*renderer.voxel_res, 3)
    return sdf, mpm_renderer.bbox_min.download(), mpm_renderer.bbox_max.download()



def lookat(center, theta, phi, radius):
    R = transforms3d.euler.euler2mat(theta, phi, 0., 'sxyz')
    b = np.array([radius, 0., 0.], dtype=float)
    back = R[0:3, 0:3].dot(b)

    p = center - back
    rot = transforms3d.quaternions.mat2quat(R)
    return Pose(p, rot)


def add_camera_light():
    camera_mount = scene.create_actor_builder().build_kinematic()
    camera = scene.add_mounted_camera(
        name="camera",
        actor=camera_mount,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=1280,
        height=720,
        fovy=np.deg2rad(45),
        near=0.1,
        far=100,
    )
    #camera_mount.set_pose(Pose(np.array([0.5, 0.5, 2.0]), np.array([0.5, -0.5, 0.5, 0.5])))
    camera_mount.set_pose(lookat(np.array([0.0, 0.0, 0.1]), 0.0, 0.8, 0.8))

    scene.set_ambient_light(np.array([0.3, 0.3, 0.3]))
    scene.add_directional_light(np.array([0, 0.5, -1]), color=np.array([3.0, 3.0, 3.0]))
    return camera


def add_ground():
    ground_material = renderer.create_material()
    ground_material.base_color = np.array([202, 164, 114, 256]) / 256
    ground_material.specular = 0.3
    ground_material.metallic = 0.0
    ground_material.roughness = 0.1

    ground_material = renderer.create_material()
    ground_material.set_base_color(np.array([0.3, 0.8, 0.8, 1.]))
    ground_material.set_diffuse_texture_from_file(os.path.join(ASSETS_PATH, 'Ground_Texture.jpg'))

    scene.add_ground(0, render_material=ground_material)



def add_softbody(pcd):
    sdf, bmin, bmax = softbody_sdf(pcd)

    vertices, triangles, _, _ = measure.marching_cubes(sdf, 0)
    vertices /= sdf.shape[-1]
    vertices = vertices * (bmax - bmin) + bmin

    builder = scene.create_actor_builder()
    material = renderer.create_material()
    material.base_color = np.array([0.749, 0.63, 0.565, 1.])
    material.roughness = 0.5
    material.metallic = 0.0

    mesh = renderer.create_mesh(vertices, triangles) # type: ignore
    builder.add_visual_from_mesh(mesh, material=material)
    builder.add_box_collision(half_size=np.array([0.1, 0.1, 0.1]))
    box = builder.build()
    box.set_pose(Pose(p=np.array([0.0, 0.0, 0.0]), q=transforms3d.euler.euler2quat(0, 0, -1)))



def render():
    scene.update_render()
    camera.take_picture()


    rgb = camera.get_color_rgba()
    rgb = im.fromarray((rgb * 255).astype(np.uint8))
    rgb.save(f'haha.png')


camera = add_camera_light()
add_ground()

def test():
    pcd = np.random.random((10000, 3)) * 0.1
    add_softbody(pcd)
    render()

    
if __name__ == '__main__':
    test()
