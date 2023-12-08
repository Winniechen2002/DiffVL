# wrap a plasticinelab env into a gym env
import os
import numpy as np
import cv2
from ui.gui import GUI, Configurable, Viewer
import gym
import sapien.core as sapien
import torch.nn.functional as F
import torch
import open3d as o3d

from pytorch3d.loss import chamfer_distance


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
from config import GymEnvConfig, PointNetConfig

def voxel_downsampling(points, voxel_size):
    """
    对点云数据进行体素栅格下采样
    :param points: 原始点云数据，维度为 (N, 3)，其中 N 为点的数量
    :param voxel_size: 体素大小
    :return: 下采样后的点云
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)
    downsampled_points = np.asarray(downsampled_point_cloud.points)

    return downsampled_points

class GymEnv(gym.Env):
    def __init__(self, config: GymEnvConfig) -> None:
        super().__init__()
        self.samples = config.samples
        self.voxel_size = config.voxel_size

        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.kappa = config.kappa
        self.zeta = config.zeta
        self.epsilon = config.epsilon

        self.config_task = config.task
        self.scene_id = config.scene_id
        self.target_scene_id = config.target_scene_id

        self.gui = CameraGUI()
        from envs import MultiToolEnv
        self.env = MultiToolEnv(sim_cfg=dict(max_steps = config.sim_max_step))
        self.target_env = MultiToolEnv(sim_cfg=dict(max_steps = config.sim_max_step))
        init_scene(self.env, 0)
        init_scene(self.target_env, 0)
        scene = Scene(self.env)
        self.gui.load_scene(scene)

        obs = self.reset()
        self.last_pos = self.env.get_obs()['pos']
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs.shape, dtype=np.uint8)
        self.action_space = self.env.action_space


    def get_obs(self):
        # return self.env.get_obs()['pos'].cpu()
        obs = self.env.get_obs()
        pos = obs['pos']
        if pos.shape[0] > self.samples*5:
            pos = torch.tensor(voxel_downsampling(obs['pos'].cpu().numpy(), self.voxel_size))
        if pos.shape[0] > self.samples:
            pos = pos[torch.randperm(pos.shape[0])[:self.samples]].cpu()
        # print(pos.shape, obs['qpos'])
        qpos = obs['qpos'].view(-1)[:3].expand(1, -1).cpu()
        return torch.concat((pos, qpos), dim = -2)

    def reset(self):
        init_scene(self.env, 0)
        scene = Scene(self.env)
        self.gui.load_scene(scene)

        task_id = self.config_task
        scene_id = self.scene_id
        target_scene_id = self.target_scene_id
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


    def step(self, action):
        obs = self.env.step(action)[0]
        target_obs = self.target_env.get_obs()
        qpos = obs['qpos'].reshape(-1)
        pos = obs['pos'].view(1, obs['pos'].shape[0], 3)
        target_pos = target_obs['pos'].view(1, target_obs['pos'].shape[0], 3)

        # 计算软体与目标位置之间的距离
        dist, _ = chamfer_distance(pos, target_pos, batch_reduction=None)
        # dist = dist.cuda()

        # 计算移动距离
        move_dist = F.pairwise_distance(self.last_pos, pos).mean()

        # 计算动作惩罚
        action_penalty = self.alpha * torch.norm(torch.abs(torch.tensor(action)), p=2)

        # 计算额外奖励
        dist_reward = torch.exp(-self.beta * dist)
        move_reward = self.gamma * move_dist if dist > self.epsilon else 0
        rigid_reward = self.zeta * torch.abs(obs['dist']).min().item()
        neg_reward = -self.kappa * dist

        # 根据距离判断是否到达目标位置
        if dist < self.epsilon:
            # 到达目标位置，获得正奖励
            reward = 1.0 + dist_reward + move_reward - action_penalty
        else:
            # 没有到达目标位置，获得与目标位置距离相关的负奖励
            reward = neg_reward + move_reward - action_penalty - rigid_reward

        # print(neg_reward , move_reward , action_penalty , rigid_reward, reward)
        # 更新上一个时间步的位置
        self.last_pos = pos.clone().detach()

        return self.get_obs(), float(reward), False, {'neg_reward': neg_reward, 'move_reward': move_reward, 'action_penalty': action_penalty, 'rigid_reward': rigid_reward, 'reward':reward, 'dist':dist}


    def render(self, mode='human'):
        if mode == 'human':
            obs = self.env.get_obs()
            pos = torch.tensor(voxel_downsampling(obs['pos'].cpu().numpy(), 1))
            print(pos.shape, obs['qpos'])
            qpos = obs['qpos'].view(-1)[:3].expand(1, -1).cpu()
            return torch.concat((pos, qpos), dim = -2)
        elif mode == 'rbga':
            img =  self.gui.capture()
            return img
        #return self.env.render('rgb_array')
