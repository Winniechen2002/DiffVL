from tools import CN
import matplotlib.pylab as plt
from diffrl.env import DifferentiablePhysicsEnv

# 1.19
"""
env = DifferentiablePhysicsEnv.parse(
    cfg_path='configs/plb3d.yml',
    sim_cfg=dict(quality=1.0, yield_stress=1000.),
    task_cfg=dict(TYPE="MoveObject", shape_metric=dict(chamfer=1., center=10.)),
    solver=dict(
        render_path='tmp',
        render_interval=50,
        max_iter=1000, lr=0.1
    ),
    **{"manipulator_cfg.size": (0.05, 0.2, 0.2)},
)
"""

env = DifferentiablePhysicsEnv.parse(
    cfg_path='configs/plb3d.yml',
    sim_cfg=dict(quality=1.),
    task_cfg=dict(
        TYPE="PressObject",
        shape_metric=dict(chamfer=0.01, grid=1.),
        contact_weight=10.,
    ),
    **{
        "manipulator_cfg": {
            "size": (0.03, 0.2, 0.2),
            "stiffness": 0.0,
        }
    }
)

env.reset()
out = env.task.reset(env)
state = env.get_state()

from diffrl.env.mpc_utils import MPC

mpc = MPC(num_iters=20)
mpc.mpc(env, T=50, render_path='tmp/mpc', lr=0.1, horizon=40, num_iters=5, inherit=True)