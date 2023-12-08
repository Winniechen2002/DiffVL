from ..mydsl import *
from ...envs import MultiToolEnv, WorldState
from .. import dsl, Scene, scene_type
from ..softbody import SoftBody
from ..library import be

env = MultiToolEnv()
mybool = dsl.bool
env.set_state(WorldState.get_empty_state(100))
scene = Scene(env)

@dsl.parse
def main(scene: scene_type):
    is_red = compose(SoftBody.color, rcurry(Eq, 'red')) # is_red = be(SoftBody.color, 'red')
    return length(lfilter(scene.get_object_list(), is_red))

#print(main(scene))
#exit(0)

print(main.pretty_print())
print(dsl.parse(main.pretty_print()).pretty_print())

print(dsl)
print(dsl.default_executor.eval(main, scene))