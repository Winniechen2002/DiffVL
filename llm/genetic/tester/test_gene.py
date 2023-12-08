from llm.pl.program import Input
from llm.tiny import *

from llm.genetic.gene import Gene

from llm.envs.test_utils import init_scene, MultiToolEnv
env = MultiToolEnv(gripper_cfg=dict(size=(0.03, 0.1, 0.1)))
init_scene(env, 0)
state = env.get_state()
scene = Scene(env)

gen = Gene.parse("reach(obj0, 0 + 5, 1)")

print(str(gen))
print(gen.serialize())
print(str(Gene.deserialize(gen.serialize())))

g2 = Gene.parse("test_func0([0, 1, 2])")
print(str(g2))
print(Gene.deserialize(g2.serialize()).serialize(mode='xx'))
print(g2.execute(dsl.default_executor, scene))