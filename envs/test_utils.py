import numpy as np
from .plb import MultiToolEnv, WorldState
from tools.utils import totensor

# sample 
def build_test_scene(id):
    if id == 0:
        from .soft_utils import rgb2int
        return WorldState.sample_shapes(method='box', center=(0.5, 0.08, 0.5), shape_args=(0.2, 0.2, 0.2), color=(255, 0, 0))
    elif id == 1:
        from .soft_utils import rgb2int
        return WorldState.sample_shapes(method='box', center=(0.5, 0.07, 0.5), shape_args=(0.3, 0.1, 0.1), color=(255, 0, 0)).switch_tools("Fingers")
    elif id == 2:
        from .soft_utils import rgb2int
        return WorldState.sample_shapes(method='box', center=(0.5, 0.14, 0.5), shape_args=(0.6, 0.05, 0.05), color=(255, 0, 0)).switch_tools("Gripper", [0.25, 0.05, 0.5, np.pi/4, 0.05])
    else:
        raise NotImplementedError


def sample_tool_trajs(tool_name, ids):
    # functions that returns initial

    def traj_Gripper_1(env:MultiToolEnv):
        q = env.tools['Gripper'].empty_state()
        q[:3] = np.array([0.2, 0.1, 0.2])
        yield q
        for i in range(50):
            yield [1., 0., 1., 0., -1. if (i//10)%2 == 1 else 1]

    return locals()[f"traj_{tool_name}_{ids}"]

def execute_traj(env: MultiToolEnv, state: WorldState, tool_name, ids=0, **kwargs):
    gen = sample_tool_trajs(tool_name, ids)(env)
    state = state.switch_tools(tool_name, next(gen))
    env.set_state(state, **kwargs)
    images = []
    for action in gen:
        env.step(totensor(action, env.device))
        images.append(env.render('rgb_array'))
    return images


def init_scene(sim: MultiToolEnv, id):
    state = build_test_scene(id)
    sim.set_state(state)

    state2  = sim.get_state()
    assert state.allclose(state2)

    
def test_copy(env:MultiToolEnv, state: WorldState):
    action = env.action_space.sample()
    env.set_state(state, requires_grad=True)
    env.step(action)
    result0 = env.get_state(20)

    #env.set_state(state, requires_grad=True)
    env._idx = 0
    env.step(action)
    result1 = env.get_state(20)
    #env.step(action)
    #result1 = env.get_state()
    assert result0.allclose(result1, return_keys=True, atol=1e-3, rtol=0.02)