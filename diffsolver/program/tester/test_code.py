# from diffsolver.program.operators.utils import add
import torch
import gc
from diffsolver.program.common import get_iobj, shape_match, touch
from diffsolver.program.temporal_ops import tlast, tkeep, tand, Constraint, AndConstraint
from diffsolver.utils.test_utils import generate_scene_traj


def test_traj_code():
    env, traj = generate_scene_traj()

    obj0 = get_iobj(0)
    obj1 = get_iobj(1)
    shape = shape_match(obj0, obj1, 0.001) 

    touch0 = touch(obj0, 0.001)

    loss, sat = shape(traj)
    assert loss.shape == ()
    assert sat.dtype == torch.bool

    loss, sat = touch0(traj)
    assert loss.shape == ()
    assert sat.dtype == torch.bool

    final =  tand(tlast(shape), tkeep(touch0))

    c = final(traj)

    assert isinstance(c, AndConstraint)
    assert len(c.elements) == 2
    assert c.loss.dtype == torch.float32
    assert c.loss.shape == ()
    assert c.sat.shape == ()
    assert c.sat.dtype == torch.bool
    assert c.loss == c.elements[0].loss + c.elements[1].loss


    del env, traj
    gc.collect()

def test_tracer():
    obj0 = get_iobj(0)
    obj1 = get_iobj(1)
    shape = shape_match(obj0, obj1, 0.001) 

    touch0 = touch(obj0, 0.001)
    assert str(shape) == 'shape'
    assert str(touch0) == 'touch0'
    assert str(shape_match(obj0, obj1, 0.001)) == 'shape_match(obj0, obj1, 0.001)'

if __name__ == '__main__':
    test_tracer()