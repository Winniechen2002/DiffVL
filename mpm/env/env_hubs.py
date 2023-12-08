import numpy as np

def comp_env(**kwargs):
    from . import DifferentiablePhysicsEnv
    env = DifferentiablePhysicsEnv.parse(
        parse_prefix = 'env', 
        cfg_path='configs/plb_cuda.yml',
        manipulator_cfg=dict(
            size=(0.03, 0.08, 0.08),
            friction=5.,
            h=0.2,
            r=0.04,
            default_type=0,
        ),
        sim_cfg=dict(
            ground_friction=5.,
            nu=0.2,
            gravity=(0., -0.2, 0.),
            **kwargs,
        ),
        observer_cfg=dict(
            center=(0.5, 0.2, 0.5),
            theta=np.pi/4,
            phi=0.,
            radius=2.
        ),
    )

    def loss_fn(idx, **kwargs):
        return kwargs['dist'].min(axis=0)[0].sum() - kwargs['pos'][:, 1].mean(axis=0)
    env._loss_fn = loss_fn

    """
    if len(env.primitives.primitives) > 2:
        env.primitives[2].h[None] = 0.4
        env.primitives[2].friction[None] = 100.
        env.primitives[2].set_state(0, [0.5, 0., 0.5, 1., 0., 0., 0.])

    env.manipulator.switch_type(0)
    """
    return env