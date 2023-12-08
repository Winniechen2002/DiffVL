# https://github.com/haosulab/sapien-vulkan-2/blob/dc505d385e8d82ae8f915e61579dd9174bcb2112/doc/shader/shader.html

# https://github.com/google/brax/blob/main/brax/io/html.py

# please install sapien at 
#TODO: reduce the control frequency ..

import os

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack
from mpm.types import vec3
from tools import Configurable

try:
    import sapien.core as sapien
except ImportError:
    print("Please install sapien from https://storage1.ucsd.edu/wheels/sapien-dev/")
    exit(0)

from .viewer import Viewer


from llm.tiny import SoftBody, Scene
import typing

ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets')


def plb2pose(p: torch.Tensor):
    assert p.shape[-1] == 7
    pos = p[..., :3]
    rot = p[..., 3:]
    from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix

    pose = pos.new(*pos.shape[:-1], 4, 4)
    pose[..., 3, 3] = 1
    pose[..., :3, 3] = pos[..., [0, 2, 1]]
    x = quaternion_to_matrix(rot)
    x = x[..., [0, 2, 1], :]
    x = x[..., :, [0, 2, 1]]
    pose[..., :3, :3] = x
    return pose



class GUI(Configurable):
    def __init__(self, cfg=None):
        super().__init__()
        shader_dir = os.path.join(os.path.dirname(__file__), "shader", "point")
        sapien.VulkanRenderer.set_camera_shader_dir(shader_dir)
        sapien.VulkanRenderer.set_viewer_shader_dir(shader_dir)

        engine = sapien.Engine()  # Create a physical simulation engine
        renderer = sapien.VulkanRenderer()  # Create a Vulkan renderer

        engine.set_renderer(renderer)  # Bind the renderer and the engine

        scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
        scene.set_timestep(1 / 100.0)  # Set the simulation frequency


        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([-0.5, -0.1, -1], [0.5, 0.5, 0.5], shadow=True)
        #scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        
        from .plugin import Reshaper, MouseTracker, ActorDragger, NormalController, ShapeMover, Recorder, Switcher, ShapeAdder, MouseDragger, ShapeDrawer , ShapeDeleter , ShapeMouseEraser, SelectDragger #, ActionReplayer
        plugins = [Reshaper(self), MouseTracker(self), ActorDragger(self), NormalController(self), ShapeMover(self), Switcher(self), ShapeAdder(self), ShapeDeleter(self), ShapeMouseEraser(self)] #, ActionReplayer(self)]


        plugins.append(Recorder(self)) #NOTE: recorder must be in the end ..
        viewer = Viewer(renderer, plugins=plugins)  # Create a viewer (window)
        viewer.gui = self
        viewer.set_scene(scene)  # Bind the viewer and the scene

        viewer.set_camera_xyz(0.5, -2.5, 1.2)
        viewer.set_camera_rpy(0, 0.0, 3.14 + 3.14/2)
        #viewer.world_space_to_camera_space
        #viewer.lookat()

        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self._scene = scene 
        self._engine = engine
        self._renderer = renderer
        self._viewer = viewer

        self.setup_environment()

        self._soft_bodies = {} # list of soft bodies
        self._tools = {}
        self.use_knife = False

        
        self._remaining_actions = None
        self.action = None

        
    def setup_environment(self):
        ground_material = self._renderer.create_material()
        ground_material.set_base_color([0.3, 0.8, 0.8, 1.])
        ground_material.set_diffuse_texture_from_file(os.path.join(ASSETS_PATH, 'Ground_Texture.jpg'))
        # self._renderer.
        self._scene.add_ground(altitude=0.0, render_material=ground_material)
        rs = self._scene.renderer_scene
        render_scene = rs._internal_scene
        render_context = self._renderer._internal_context

        #print(render_scene)
        points = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
        lineset = [[], []]
        for i in range(len(points)):
            lineset[0] += points[i]
            lineset[0] += points[(i + 1) % len(points)]
            lineset[1] += [0.9254901960784314, 0.5764705882352941, 0.18823529411764706, 1] * 2


        #print(np.array(lineset[0]).shape, np.array(lineset[1]).shape)
        #exit(0)
        lineset = render_context.create_line_set(*lineset)
        self.boundary = render_scene.add_line_set(lineset)


    def _add_sapien_particles(self, softbody: SoftBody):
        N = softbody.N()

        scales = np.ones(N, dtype=np.float32) * 0.012
        # colors = np.zeros((N, 4)) + softbody.rgba()/255. #np.array([0.8, 0.8, 0.2, 0.0])
        # print(list(np.zeros((N, 4)) + softbody.rgba()/255.))
        colors = softbody.get_color()/255.
        colors = colors.astype(np.float32)
        # print(list(colors))
        mylabel = np.zeros((N, 3), dtype=np.float32) 
        mylabel[:, 0] = softbody.object_id

        pcd = self._scene.add_particle_entity(np.zeros((N, 3), dtype=np.float32))

        pcd.visual_body.set_attribute("color", colors)
        pcd.visual_body.set_attribute("scale", scales)
        pcd.visual_body.set_attribute("mylabel", mylabel.astype(np.float32))

        particle_tensor = from_dlpack(pcd.visual_body.dl_vertices)
        particle_tensor[:, 7] = 0.0001
        
        elements = {
            'pcd': pcd, 
            'tensor': particle_tensor, 
            'indices': softbody.indices,
            'N': N, 
            'id': softbody.object_id, 
            'colors':  colors
        }
        # print(elements)
        # print(elements['pcd'])
        # print(elements['tensor'].shape, elements['indices'].shape, elements['N'], elements['id'], elements['colors'].shape)

        self._soft_bodies[softbody.object_id] = elements
        return elements


    def selected_tool(self):
        #TODO: this is not efficient ..
        viewer = self._viewer
        if viewer.selected_entity:
            for k in self._tools.values():
                if k == viewer.selected_entity:
                    return True
        return False
                    
        
    def _add_sapien_tools(self, tool_name):
        #print("_add_sapien_tools",tool_name)
        # NOTE: we do not support change the size now ..
        if len(self._tools) > 0:
            if self._viewer is not None:
                self._viewer.select_entity(None)
            for k, v in self._tools.items():
                self._scene.remove_actor(v)
            self._tools = {}

        tool_cfg = self._env.tools[tool_name]._cfg
        self.tool_name = tool_name

        def add_object(cfg, name):
            if self.use_knife:
                builder = self._scene.create_actor_builder()
                assert isinstance(self._renderer, sapien.VulkanRenderer)
                scale = 0.03
                builder.add_visual_from_file(os.path.join(ASSETS_PATH, 'Meat_Cleaver.obj'), scale=np.array([scale, scale, scale]), pose=sapien.Pose([-0.03, 0.25, 0.05], [0.5, 0.5, 0.5, 0.5]))
                knife = builder.build(name=name)
                return knife
            elif cfg.mode == 'Box':
                builder: sapien.ActorBuilder = self._scene.create_actor_builder()
                x, z, y = cfg.size
                half_size = (x, y, z)
                material = self._renderer.create_material()
                if name == 'Gripper0' or name == 'Gripper1':   
                    material.set_diffuse_texture_from_file(os.path.join(ASSETS_PATH, 'WoodTexture.jpg'))
                    material.roughness = 1.
                    material.metallic = 0.0
                elif name == 'Knife' or self.use_knife:
                    material.set_diffuse_texture_from_file(os.path.join(ASSETS_PATH, 'knife.jpeg'))
                    material.roughness = 1.
                    material.specular = 0.5
                    # material.metallic = 1.0
                else:
                    material.roughness = 1.
                    material.metallic = 0.0
                    material.set_diffuse_texture_from_file(os.path.join(ASSETS_PATH, 'WoodTexture.jpg'))
                builder.add_box_visual(half_size=half_size, material=material)  # Add visual shape
                #name = f'Gripper{i}'
                box: sapien.Actor = builder.build(name=name)
                return box
            elif cfg.mode == 'Capsule':
                builder: sapien.ActorBuilder = self._scene.create_actor_builder()
                r, length = tool_cfg.size

                material = self._renderer.create_material()
                if name == 'Rolling_Pin':
                    material.base_color = [255./255, 211./255, 155./255, 1.0]
                    material.roughness = 1.
                    material.metallic = 0.0
                    material.set_diffuse_texture_from_file(os.path.join(ASSETS_PATH, 'Texture.png'))

                from transforms3d.quaternions import axangle2quat
                builder.add_capsule_visual(pose=sapien.Pose(
                    [0, 0., length/2], axangle2quat([0, 1, 0], np.pi/2)), 
                    radius=r, half_length=length/2, material = material)
                capsule: sapien.Actor = builder.build(name=name)
                return capsule
            else:
                raise NotImplementedError(f'Unknown tool mode {cfg.mode}')

        if tool_name == 'Gripper':
            for i in range(2):
                name = f'Gripper{i}'
                self._tools[name] = add_object(tool_cfg, name)
        
        elif tool_name == 'DoublePushers':
            for i in range(2):
                name = f'Pusher{i}'
                self._tools[name] = add_object(tool_cfg, name)

        elif tool_name == 'Pusher':
            name = 'Pusher'
            self._tools[name] = add_object(tool_cfg, name)

        elif tool_name == 'Knife':
            name = 'Knife'
            self._tools[name] = add_object(tool_cfg, name)

        elif tool_name == 'Rolling_Pin':
            name = 'Rolling_Pin'
            self._tools[name] = add_object(tool_cfg, name)

        else:
            raise NotImplementedError


    def load_scene(self, scene):

        for k, v in self._soft_bodies.items():
            self._scene.remove_particle_entity(v['pcd'])

        for k, v in self._tools.items():
            self._scene.remove_actor(v)

        scene: Scene
        self.last_scene = scene
        self._env = scene.env
        self._soft_bodies = {}
        object_list: typing.List[SoftBody] = scene.get_object_list()

        for i in object_list:
            self._add_sapien_particles(i)

        self._add_sapien_tools(self._env.tool_cur.name)

    def reload_scene(self, scene):
        for k, v in self._soft_bodies.items():
            self._scene.remove_particle_entity(v['pcd'])
        # print('Init', self._soft_bodies)
        scene: Scene
        self.last_scene = scene
        self._env = scene.env
        self._soft_bodies = {}
        object_list: typing.List[SoftBody] = scene.get_object_list()
        # print('object_list', object_list)

        for num, i in enumerate(object_list):
            self._add_sapien_particles(i)
        self.update_scene()

    
    def update_scene(self):
        self.obs = self._env.get_obs()

        _tensor = self.obs['pos']
        for k, v in self._soft_bodies.items():
            # print(_tensor.shape)
            # print(v)
            p = _tensor[v['indices']]#.detach().cpu().numpy()
            p = p[:, [0, 2, 1]]
            v['tensor'][:, :3] = torch.tensor(p, device='cuda:0') #torch.rand((5000, 3), device='cuda:0', dtype=torch.float32) * 0.2 

        pose = plb2pose(self.obs['tool']).detach().cpu().numpy()

        for p, actor in zip(pose, self._tools.values()):
            import transforms3d
            pos = p[:3, 3]
            rot = transforms3d.quaternions.mat2quat(p[:3,:3])
            actor: sapien.Actor
            actor.set_pose(sapien.Pose(pos, rot))


    def step(self):
        # step MPM
        assert self._env is not None, "please first load a scene .."
        if self._remaining_actions is None:
            action = self._env.action_space.sample() * 0 
        else:
            action = self._remaining_actions.pop(0)
            if len(self._remaining_actions) == 0:
                self._remaining_actions = None


        velocity = None

        for plugin in self._viewer.plugins:
            action, _ = plugin.step(action)
            if not _ is None:
                if velocity is None:
                    velocity = _
                else:
                    velocity += _

        self._action = action
        # print(velocity)

        self._env.step(action, velocity)

        self.action = None
        self._scene.step()  # Simulate the world

    def update_scene_by_state(self, state):
        env = self._env
        scene = Scene(env, collect_obs=False)
        env.set_state(state)
        scene.collect(0)
        scene.initial_state = state
        scene.env.initial_state = state
        self.reload_scene(scene) # update based on scene ..
            
    def start(self):
        import time
        target_fps = 60
        while not self._viewer.closed:  # Press key q to quit
            start_time = time.time()

            self.step()
            if time.time() - start_time < 1./target_fps:
                time.sleep(max(1./target_fps - (time.time() - start_time), 0))

            self._scene.update_render()  # Update the world to the renderer

            self.update_scene() # update the shape ..

            self._viewer.render()
