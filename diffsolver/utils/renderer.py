import os
import torch
import numpy as np
from typing import Any
import cv2
from ui.gui import GUI, Configurable, Viewer
import gym
import sapien.core as sapien
from llm.tiny import SoftBody, Scene
import torch.nn.functional as F

SIM_RENDERER = None
def create_sim():
    global SIM_RENDERER
    if SIM_RENDERER is None:
        from mpm.simulator import MPMSimulator
        from mpm.renderer import Renderer
        simulator = MPMSimulator(1, n_particles=100000, max_steps=10)
        #mpm_renderer = Renderer(voxel_res=(256, 256, 256))
        mpm_renderer = Renderer()
        SIM_RENDERER = (simulator, mpm_renderer)
    return SIM_RENDERER

class CameraGUI(GUI):
    """
    render plb using sapien camera
    """

    def __init__(self, cfg=None, offscreen=False, camera_config = None, ray_tracing = 0):
        Configurable.__init__(self)
        self.use_knife = False

        shader_dir = os.path.join(os.path.dirname(__file__), "shader", "point")
        self.ray_tracing = ray_tracing

        if ray_tracing:
            sapien.render_config.camera_shader_dir = "rt"
            sapien.render_config.viewer_shader_dir = "rt"
            sapien.render_config.rt_samples_per_pixel = ray_tracing  # change to 256 for less noise
            sapien.render_config.rt_use_denoiser = False  # change to True for OptiX denoiser
            renderer = sapien.VulkanRenderer(offscreen_only=True)  # Create a Vulkan renderer
        else:
            sapien.VulkanRenderer.set_camera_shader_dir(shader_dir)
            sapien.VulkanRenderer.set_viewer_shader_dir(shader_dir)
            renderer = sapien.SapienRenderer(offscreen_only=True)  # Create a Vulkan renderer


        engine = sapien.Engine()  # Create a physical simulation engine

        engine.set_renderer(renderer)  # Bind the renderer and the engine

        scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
        scene.set_timestep(1 / 100.0)  # Set the simulation frequency


        if ray_tracing == 0:
            scene.set_ambient_light(np.array([0.5, 0.5, 0.5]))
            scene.add_directional_light(np.array([-0.2, -0.5, -1]), np.array([0.5, 0.5, 0.5]), shadow=True)
        else:
            scene.set_ambient_light(np.array([0.5, 0.5, 0.5]))
            #scene.add_directional_light(np.array([0, 0.0, -1.]), color=np.array([1.0, 1.0, 1.0]), shadow=True, scale=2.0, shadow_map_size=4096)
            scene.add_point_light(np.array([0, 0.0, 1.]), color=np.array([1.0, 1.0, 1.0]))
        #scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        self._scene = scene 
        self._engine = engine
        self._renderer = renderer
        if camera_config:
            self.setup_camera(camera_config.pos, camera_config.quat)
        else:
            self.setup_camera()


        self._soft_bodies = {} # list of soft bodies
        self._render_mesh = {}
        self._tools = {}

        
        self._remaining_actions = None
        self.action = None
        self.setup_environment()

        if not offscreen:
            self.setup_viewer()
        else:
            self._viewer = None


    def setup_camera(self, pose = [0.5, 0.5, 2.], quat = [0.5, -0.5, 0.5, 0.5]):

        near, far = 0.1, 100
        width, height = 512, 512
        camera = self._scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )
        print(pose)
        camera.set_pose(sapien.Pose(p=np.array(pose), q=np.array(quat)))
        self._camera = camera

    def lookat(self, cfg):
        from tools.utils import lookat # optimize import speed ..
        import transforms3d

        R = transforms3d.euler.euler2mat(cfg.theta, cfg.phi, cfg.zeta, 'sxyz')
        b = np.array([cfg.radius, 0., 0.], dtype=float)
        back = R[0:3, 0:3].dot(b)
        print(cfg.center, back)
        self.setup_camera(cfg.center - back, transforms3d.quaternions.mat2quat(R))
        
    def setup_viewer(self):
        from ui.plugin import Reshaper, NormalController, ShapeMover, Recorder, Switcher, ShapeAdder, MouseDragger, ShapeDrawer , ShapeDeleter , ShapeMouseEraser, SelectDragger #, ActionReplayer
        plugins = [Reshaper(self), NormalController(self), ShapeMover(self), Switcher(self), ShapeAdder(self), MouseDragger(self), ShapeDeleter(self), ShapeMouseEraser(self), SelectDragger(self)] #, ActionReplayer(self)]


        plugins.append(Recorder(self)) #NOTE: recorder must be in the end ..
        viewer = Viewer(self._renderer, plugins=plugins)  # Create a viewer (window)
        viewer.gui = self # type: ignore
        viewer.set_scene(self._scene)  # Bind the viewer and the scene

        viewer.set_camera_xyz(0.5, -2.5, 1.2)
        viewer.set_camera_rpy(0, 0.0, 3.14 + 3.14/2)

        assert viewer.window is not None
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self._viewer = viewer


    def capture(self):
        # if objects is not None:
        #     for k in objects:
        #         self._add_pcd(**k)
        self._scene.step()
        self.update_scene() # update the shape ..
        self._scene.update_render()

        if self._viewer is not None:
            assert self._viewer.coordinate_axes is not None
            self._viewer.coordinate_axes.set_position([100, 0, 0]) # move away the axis

        self._camera.take_picture()

        rgba = self._camera.get_float_texture("Color")
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        return rgba_img

    
    def reset(self, env):
        from llm.tiny import Scene
        scene = Scene(env)
        self.load_scene(scene)

    def _add_sapien_particles(self, softbody: SoftBody):
        if not self.ray_tracing:
            return super()._add_sapien_particles(softbody)
        import torch
        from skimage import measure

        pcd = softbody.pcd()
        colors = softbody.get_color()[:, :3]
        return self._add_pcd(pcd, colors, object_id=softbody.object_id)


    def _add_pcd(self, pcd: torch.Tensor, colors: np.ndarray, object_id: Any='', transmission=0.):
        import torch
        from skimage import measure
        N = len(pcd)
        pcd = pcd.clone()
        pcd[:, 1] -= 0.03

        #from .kuafu import softbody_sdf, measure
        simulator, mpm_renderer = create_sim()

        simulator.states[0].x.upload(pcd.detach().cpu().numpy())

        simulator.n_particles = N
        img = mpm_renderer.render(simulator, simulator.states[0])
        sdf = mpm_renderer.sdf_volume.download().reshape(*mpm_renderer.voxel_res) - 0.414
        bmin, bmax = mpm_renderer.bbox_min.download(), mpm_renderer.bbox_max.download()

        vertices, triangles, normal, _ = measure.marching_cubes(sdf, 0.0)
        triangles = triangles[:, [0, 2, 1]]
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        # Compute the normals
        mesh.compute_vertex_normals()
        normal = np.asarray(mesh.vertex_normals)


        K = len(vertices)
        # vertices_color = color_volume[v[:, 0], v[:, 1], v[:, 2]]

        vertices /= sdf.shape[-1]
        vertices = vertices * (bmax - bmin) + bmin
        vertices = vertices[..., [0, 2, 1]]


        from pytorch3d.ops import knn_points
        nearest = knn_points(torch.tensor(vertices).float().cuda()[None,:], pcd[None,:], K=1)[1][0, :, 0]
        nearest = nearest.cpu().numpy()
        vertices_color = colors[nearest]
        # print(colors[0], vertices_color[0], colors.mean(axis=0))
        print(len(pcd), len(vertices), len(vertices_color), vertices_color.mean(axis=0), colors.mean(axis=0))

        
        builder = self._scene.create_actor_builder()
        material = self._renderer.create_material()
        material.base_color = np.array([1., 1., 1., 1.])
        #material.base_color = np.array([1.0, 1.0, 1.0, 1.])
        material.roughness = 0.5
        material.specular = 0.0
        material.metallic = 0.0
        # material.transmission = transmission

        c = np.zeros((K, 1, 4), dtype=np.uint8)
        c[:, 0, :3] = vertices_color
        c[:, :, -1] = int(255 * (1-transmission))
        # c = np.float32(c/255.)
        #texture = self._renderer.create_texture_from_array(c)
        
        texture = self._renderer.create_texture_from_array(c, filter_mode='nearest')
        material.set_diffuse_texture(texture)

        assert len(vertices) == len(vertices_color)
        mesh = self._renderer.create_mesh(vertices, triangles) # type: ignore
        uvs = np.float32(np.concatenate((np.arange(K).reshape(-1, 1), np.zeros((K, 1))), -1))
        print(vertices_color.mean(axis=0))

        assert normal.shape == (K, 3)
        mesh.set_normals(normal) # type: ignore
        mesh.set_uvs(uvs) # type: ignore


        builder.add_visual_from_mesh(mesh, material=material)
        obj = builder.build()
        
        particle_tensor = torch.tensor(pcd).float()
        elements = {
            'pcd': obj, 
            'tensor': particle_tensor, 
            'N': N, 
        }
        #self._soft_bodies[softbody.object_id] = elements
        print(object_id, N)
        self._render_mesh[object_id] = elements
        return elements

    def load_scene(self, scene):
        for k in self._render_mesh:
            self._scene.remove_actor(self._render_mesh[k]['pcd'])
        self._renderer_mesh = {}
        super().load_scene(scene)