# running sb3 to solve some tasks
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from sac import SAC
import gym

from gym import spaces
from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from pointnet import PointNetfeat

import argparse
import random
import math

import wandb

from config import GymEnvConfig, PointNetConfig
from omegaconf import OmegaConf

def get_args():
    conf = OmegaConf.load("PLBRL/assets/config.yaml")
    conf.gym_env = OmegaConf.structured(GymEnvConfig(**conf.gym_env))
    conf.pointnet = OmegaConf.structured(PointNetConfig(**conf.pointnet))
    return conf


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class PointNet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, config: PointNetConfig):
        super().__init__(observation_space, config.features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.PointNet = PointNetfeat(config.hidden_size)

        # Compute shape by doing one forward pass

        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size + 3, config.hidden_size), nn.ReLU(),
            nn.Linear(config.hidden_size, config.features_dim), nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs = observations[:,:-1,:]
        qpos = observations[:,-1,:].view(-1,3)
        obs = obs.view(obs.shape[0],obs.shape[2],-1)
        # print(obs.shape)
        out = self.PointNet(obs)
        # print(out.shape, qpos.shape)
        out = torch.concat((out , qpos), dim = -1)
        return self.linear(out)

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )

from wandb_callback import *


def run_sb3():

    args = get_args()

    wandb.init(config=args,
               project = 'TaskAnnotator',
               entity = 'taskannotator',
               name = args.name,
               job_type = "training",
               reinit=True)

    gym.envs.register(
        'MoveTest-v0',
        entry_point='gym_env_point_cloud:GymEnv',
        max_episode_steps=args.gym_env.sim_max_step,
        kwargs={"config": args.gym_env},
    )
    policy_kwargs = dict(
        features_extractor_class=PointNet,
        features_extractor_kwargs={"config": args.pointnet},
    )
    # model = SAC("CnnPolicy", "MoveTest-v0", policy_kwargs=policy_kwargs, verbose=1)
    model = SAC("MlpPolicy", "MoveTest-v0", policy_kwargs=policy_kwargs, 
                verbose=args.verbose, tensorboard_log="./tensorboard/",
                # learning_rate = 1e-6, 
                gamma = 0.95)

    reward_length_callback = RewardLengthCallback()
    wandb_callback = WandbCallback(args.check_freq, reward_length_callback)
    eval_callback = CustomEvalCallback(model.get_env(), 
                        eval_freq=args.eval_freq, sample_steps = args.gym_env.sim_max_step, 
                        video_output_path = args.output + '.mp4')

    # wandb.log({f"reward/reward": 0}, step = 0)
    model.learn(args.steps, callback=[reward_length_callback, wandb_callback, eval_callback], log_interval=args.log_interval)
    # for _ in range(int(1e4/args.steps)):
    #     # model.learn(args.steps)
    #     # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    #     # Enjoy trained agent
    #     vec_env = model.get_env()
    #     obs = vec_env.reset()
    #     images = []
    #     for i in range(args.sample_steps):
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, rewards, dones, info = vec_env.step(action)
    #         images.append(vec_env.render(mode = 'rbga'))
    #     from tools.utils import animate
    #     animate(images, filename = args.output + '.mp4')
        # video = wandb.Video(args.output + '.mp4', fps=30)
        # wandb.log({'my_video': video})


if __name__ == '__main__':
    run_sb3()
