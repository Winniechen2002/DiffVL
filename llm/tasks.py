import torch
import numpy as np

TASKS = dict()

def register_task(name=None):
    def wrapper(fn):
        _name = name
        if _name is None:
            _name = str(fn.__name__)
        TASKS[_name] = fn
        return fn

    if callable(name):
        fn = name
        name = None
        return wrapper(fn)
    return wrapper

    
@register_task
def Move_v0(env, cfg=None):
    from .envs.test_utils import init_scene
    init_scene(env, 0)
    state = env.get_state()
    state = state.switch_tools('Gripper', np.array([0.5, 0.1, 0.5, 0., 0.2]))
    
    def loss_fn(iter, **obs):
        lift = - obs['pos'].mean(axis=0)[1]
        contact = torch.relu(obs['dist'].min(axis=0).values).sum()
        return lift + contact, {'lift': lift.item(), 'contact': contact.item()}
    return state, loss_fn


def build_task(env, task_name, task_cfg):
    # a task is a pair of initial state and loss_fn
    state, loss_fn = TASKS[task_name.replace('-', '_')](env, cfg=task_cfg)
    return state, loss_fn