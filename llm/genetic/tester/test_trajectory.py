import matplotlib.pyplot as plt
from llm.tiny import Scene, scene_type, dsl
from llm.envs import MultiToolEnv
from llm.genetic.trajectory import *
from llm.genetic.libraries.init_state import str2state

myint = dsl.int


init_state = str2state('box.yml')

env = MultiToolEnv(sim_cfg=dict(max_steps=1024, n_particles=5000, ground_friction=0.8))
env.set_state(init_state)

scene = Scene(env, init=init_state)
for i in range(100):
    env.step(env.action_space.sample())
    scene.collect()


@dsl.as_func
def add(a: myint, b: myint):
    return a + b

@dsl.parse
def main(scene: scene_type):
    obj = scene.obj(0)
    x = add(None, 1)
    scene.new_stage(25).keep(lambda: car(obj.pcd().com()) < 0.).execute()
    scene.new_stage(25).keep(lambda: obj.t() < 51).execute()
    return x(1)
    
from llm.pl.program import set_comment
set_comment(False)
print(main.pretty_print())

print(main(scene))
print(scene.tables)
main(scene)
print(scene.tables)