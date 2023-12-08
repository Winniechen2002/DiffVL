from .prog_registry import register_prog
from .temporal_ops import tkeep, tlast, tand
from .common import *


@register_prog('lift')
def lift():
    obj0 = get_iobj(0)
    return tand(tlast(above(obj0, 0.3)), tkeep(touch(obj0, 0.02), weight=5.))


@register_prog('pickup_rope')
def pickup_rope():
    rope = get_iobj(1)
    return tand(tlast(above(rope, 0.3)), tkeep(touch(rope, 0.02)))


@register_prog('wind_rope')
def wind_rope():
    cylinder = get_iobj(0)
    rope = get_iobj(1)
    return tand(tlast(shape_match(rope, cylinder, 0.01), weight=20.), tkeep(touch(rope, 0.02), weight=0.1))

    
    
@register_prog('task10')
def task10():
    obj0 = get_iobj('top_left_mat')
    goal = get_goal('right_core')
    return tand(tlast(shape_match(obj0, goal, 0.01)), tkeep(touch(obj0, 0.02)))
