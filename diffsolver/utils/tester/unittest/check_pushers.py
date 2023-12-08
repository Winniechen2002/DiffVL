from tools.utils import animate
from diffsolver.config import SceneConfig
from diffsolver.program.scenes import load_scene_with_envs
from diffsolver.utils import MultiToolEnv

env = MultiToolEnv()

scene = load_scene_with_envs(
    env,
    SceneConfig(
        path='1_2_2.task',
        Tool = dict(
            tool_name =  'DoublePushers',
            qpos = [0.25, 0.2, 0.5, 0., 0., 0., 0.75, 0.2, 0.5, 0., 0., 0.],
            size = (0.03, 0.2, 0.2),
        )
        
    )
)


env.set_state(scene.state)
print(env.action_space)
images = []
for i in range(100):
    env.step(env.action_space.sample())
    img = env.render('rgb_array')
    assert img is not None
    images.append(img)

animate(images)