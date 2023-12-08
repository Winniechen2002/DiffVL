from .gym_env import make, GymEnv
from gym.envs.registration import register


register(
    'Lift-v0',
    entry_point='diffsolver.rl_baselines.rl_envs:make',
    kwargs={"config_path": 'lift.yml'},
)