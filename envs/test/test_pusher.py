import tqdm
from envs.plb import MultiToolEnv
from tools.utils import animate

env = MultiToolEnv()

from envs.test_utils import init_scene

init_scene(env, 0)
state = env.get_state()
state = state.switch_tools("Pusher", qpos=[0., 0., 0.])


env.set_state(state)
s = env.get_state()

images = []
for i in range(100):
    env.step(env.action_space.sample() * 0)
    images.append(env.render('rgb_array'))
    print(images[-1].shape)

animate(images)
