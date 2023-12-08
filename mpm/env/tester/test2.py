from tools import CN
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
env.execute(
    env.get_state(),
    [[1., 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0]for i in range(4)] +
    [[-1., 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]for i in range(10)],
    'animation.mp4'
)

# env.simulator.penalty_force_scale[None] = 1.0
env.task.reset(env)

out = env.solve(env.task.reset(env),
                [env.manipulator.get_initial_action() for i in range(50)],
                render_path='tmp2',
                render_interval=50,
                max_iter=1000)
