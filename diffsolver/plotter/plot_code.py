from diffsolver.program import * 
from diffsolver.program.common import * 
from diffsolver.program.progs import * 
from diffsolver.program.progs import * 
from diffsolver.toolguide.tool_progs.equations import * 
from diffsolver.toolguide.tool_progs.constraints import * 
from diffsolver.program.temporal_ops import last, keep

def optimize(*args):
    pass

def stage(*args):
    pass


def close(*args):
    pass

def backend(*args):
    pass


def frontend(*args):
    pass


def leftpart(*args):
    pass


def sample(
    *args, **kwargs
):
    pass


def l2(*args):
    pass

def y(*args):
    return 1

def ensur(
    *args, **kwargs
):
    pass

def backend(*args):
    pass

def frontend(*args):
    pass

def grasp(*args):
    pass

def require(*args):
    pass

def roll(*args):
    return 0.

def pitch(*args):
    return 0.

def keep(*args):
    pass


sample
sample("gripper", grasp(backend(get("rope"))))
optimize(require(emd(get("rope"), goal("knife"))))
sample("gripper", grasp(frontend(get("rope"))))
optimize(require(emd(get("rope"), goal("knife"))))


sample
sample("gripper", grasp(get("white")))
optimize(
    require(y(com('brown')) > y(max(pcd('black'))), 0.5), 
    require(l2(com(get("white")), com(goal("white")))),
)

sample
sample("knife", roll(0.), y('knife') > y(max(pcd('white'))))
optimize(
    keep(touch(leftpart('white'))),
    keep(roll() < 0.1), keep(pitch() < 0.1),
    require(emd('obj', goal('obj'))),
    keep(no_break('white')),
)


sample

sample("RollingPin", y('RollingPin') > y(max(pcd('white'))))
optimize(
    keep(touch('right white')),
    require(emd('right white', goal('white mat')),
)