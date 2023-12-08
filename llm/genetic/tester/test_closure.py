from llm.pl.program import Input
from llm.tiny import *


def obj1(scene: scene_type, t: myint):
    return scene.obj(0).at(t)

# dsl.closure = {'scene': Input('scene', scene_type), 't': myint}
obj1 = dsl.parse(obj1)


def cond():
    #return scene.obj(0).at(t)
    return [obj1, obj1]

cond = dsl.parse(cond, context={'obj1': obj1.value})

#print(cond.closure)


from llm.envs.test_utils import init_scene, MultiToolEnv
env = MultiToolEnv(gripper_cfg=dict(size=(0.03, 0.1, 0.1)))
init_scene(env, 0)
state = env.get_state()
scene = Scene(env)

print(dsl.default_executor._execute(cond.value, context={'scene': scene, 't': 0}))