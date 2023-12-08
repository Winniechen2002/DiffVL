# wrap a plasticinelab env into a gym env
import os
import numpy as np
import cv2
from ui.gui import GUI, Configurable, Viewer
import gym
import sapien.core as sapien
import torch.nn.functional as F


class CameraGUI(GUI):
    """
    render plb using sapien camera
    """

    def __init__(self, cfg=None):
        Configurable.__init__(self)

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

        self._scene = scene 
        self._engine = engine
        self._renderer = renderer
        self.setup_camera()


        self._soft_bodies = {} # list of soft bodies
        self._tools = {}

        
        self._remaining_actions = None
        self.action = None
        self.setup_environment()
        self.setup_viewer()

    def setup_camera(self):

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
        camera.set_pose(sapien.Pose(p=[0.5, -2, 0.2], q=[0.707, 0., 0., 0.707]))
        self._camera = camera


    def setup_viewer(self):
        from ui.plugin import Reshaper, NormalController, ShapeMover, Recorder, Switcher, ShapeAdder, MouseDragger, ShapeDrawer , ShapeDeleter , ShapeMouseEraser, SelectDragger #, ActionReplayer
        plugins = [Reshaper(self), NormalController(self), ShapeMover(self), Switcher(self), ShapeAdder(self), MouseDragger(self), ShapeDeleter(self), ShapeMouseEraser(self), SelectDragger(self)] #, ActionReplayer(self)]


        plugins.append(Recorder(self)) #NOTE: recorder must be in the end ..
        viewer = Viewer(self._renderer, plugins=plugins)  # Create a viewer (window)
        viewer.gui = self
        viewer.set_scene(self._scene)  # Bind the viewer and the scene

        viewer.set_camera_xyz(0.5, -2.5, 1.2)
        viewer.set_camera_rpy(0, 0.0, 3.14 + 3.14/2)
        #viewer.world_space_to_camera_space
        #viewer.lookat()

        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self._viewer = viewer



    def capture(self):
        self._scene.step()
        self.update_scene() # update the shape ..
        self._scene.update_render()
        self._viewer.coordinate_axes.set_position([100, 0, 0]) # move away the axis
        self._camera.take_picture()

        rgba = self._camera.get_float_texture("Color")
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        return rgba_img
        
        
from envs.test_utils import init_scene
from llm.tiny import Scene
from frontend import DATA_PATH
import pickle
class GymEnv(gym.Env):
    def __init__(self, scene_id = 0) -> None:
        super().__init__()
        self.gui = CameraGUI()
        from envs import MultiToolEnv
        self.env = MultiToolEnv(sim_cfg=dict(max_steps=100))
        self.target_env = MultiToolEnv(sim_cfg=dict(max_steps=100))
        init_scene(self.env, 0)
        init_scene(self.target_env, 0)
        scene = Scene(self.env)
        self.gui.load_scene(scene)

        obs = self.reset()
        self.last_pos = self.env.get_obs()['pos']
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs.shape, dtype=np.uint8)
        self.action_space = self.env.action_space

    def get_obs(self):
        return cv2.resize(self.render(), (84, 84))

    def reset(self):
        init_scene(self.env, 0)
        scene = Scene(self.env)
        self.gui.load_scene(scene)

        task_id = 9
        scene_id = 1
        target_scene_id = 2
        if task_id is not None:
            pkl_path = os.path.join(DATA_PATH, 'task_{}'.format(task_id))
            path = os.path.join(pkl_path, 'scene_{}.pkl'.format(scene_id))
            with open(path, 'rb') as f:
                print('load scene from', path)
                state = pickle.load(f)
            self.env.set_state(state.state)
            scene = Scene(self.env)
            self.gui.load_scene(scene)

            path = os.path.join(pkl_path, 'scene_{}.pkl'.format(target_scene_id))
            with open(path, 'rb') as f:
                print('load scene from', path)
                state = pickle.load(f)
            self.target_env.set_state(state.state)

        return self.get_obs()


    def step(self,action):
        obs = self.env.step(action)[0]
        # print(action)
        target_obs = self.target_env.get_obs()
        # print(obs['qpos'])
        qpos = obs['qpos'].reshape(-1)
        # print(obs , target_obs)
        reward = (F.pairwise_distance(obs['pos'], target_obs['pos'] , p = 2).mean()*20  
                       + F.pairwise_distance(qpos[:3], obs['pos'] , p = 2).min())
        if F.pairwise_distance(self.last_pos , obs['pos']).max() > 0.001:
            # print(self.last_pos != obs['pos'])
            reward = -float(reward) + 2
        else :
            reward = -float(reward)
        # if action[[3,4,5,9,10,11]].max() > 0.01 or action[[3,4,5,9,10,11]].min() < -0.01:
            # reward -= 500
        self.last_pos = obs['pos']
        return self.get_obs(), reward, False, {}


    def render(self, mode='human'):
        img =  self.gui.capture()
        return img
        #return self.env.render('rgb_array')
