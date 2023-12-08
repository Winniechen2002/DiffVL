from tools import CN
import matplotlib.pylab as plt
from diffrl.env import DifferentiablePhysicsEnv

# 1.19

env = DifferentiablePhysicsEnv.parse(
    cfg_path='configs/plb_cuda.yml',
    sim_cfg=dict(yield_stress=1000., n_particles=10000),
    #cfg_path='configs/plb3d.yml',
    #sim_cfg=dict(n_particles=2000),
    #use_taichi=True,

    task_cfg=dict(TYPE="MoveObject", shape_metric=dict(chamfer=1., center=10.)),
    solver=dict(
        render_path='tmp',
        render_interval=50,
        max_iter=1000, lr=0.1
    ),
    **{"manipulator_cfg.size": (0.05, 0.2, 0.2)},
)



import tqdm
import numpy as np
state = env.empty_state(n=10000)
env.set_state(state)
#for i in tqdm.trange(100000):
#    env._step(np.zeros(12))

env.reset()
print('reset')
plt.imshow(env.render('rgb_array'))
plt.savefig('tmp/init.png')

plt.imshow(env.task.render_goal('rgb_array'))
plt.savefig('tmp/goal.png')
out = env.solve(env.task.reset(env),
                [env.manipulator.get_initial_action() for i in range(50)])
