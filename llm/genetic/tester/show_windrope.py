import torch
import cv2
from llm.envs import MultiToolEnv, MPMSimulator

env_cfg=MultiToolEnv.get_default_config(
sim_cfg=dict(max_steps=100, gravity=(0., -0.9, 0.), ground_friction=1., n_particles=10000),
)
env = MultiToolEnv(cfg=env_cfg)

for i in range(4):
    a = torch.load(f'windrope/{i}.th')
    a = a.switch_tools('Pusher')
    env.set_state(a)
    img = env.render('rgb_array', primitive=False)[100:-100, 100:-100]
    cv2.imwrite(f'{i}.png', img[..., ::-1])