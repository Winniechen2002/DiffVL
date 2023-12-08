import tqdm
import numpy as np
from diffrl.env import DifferentiablePhysicsEnv

env = DifferentiablePhysicsEnv.parse(
    # cfg_path='configs/plb3d.yml',
    sim_cfg=dict(n_particles=5000),
    task_cfg=dict(
        TYPE="PressObject",
        shape_metric=dict(chamfer=0.01, grid=1.),
        contact_weight=10.,
    ),
    solver=dict(lr=0.01),
    **{
        "manipulator_cfg": {
            "size": (0.03, 0.2, 0.2),
            "stiffness": 0.0,
        }
    },
)


env.reset()
for i in tqdm.trange(1000000):
    env.step(np.zeros(12))