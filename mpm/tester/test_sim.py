from mpm.cuda_env import CudaEnv
import os

from mpm.tester.tester import generate_state, assert_state, test_p2g
import numpy as np

state = generate_state(5000)

cu_env = CudaEnv(PRIMITIVES=[{'shape': 'Box', 'size': (0.1, 0.1, 0.1), 'round': 0.2} for i in range(2)], SIMULATOR=dict(n_particles=5000, max_steps=1024))

sim = cu_env.simulator


"""
import pickle
with open('mat_tmp.pkl', 'rb') as f:
    state[2][:] = pickle.load(f)
"""


#sim.set_state(0, state)
#assert_state(sim.get_state(0), state)


"""
print(sim.dt)
sim.dt = float(0)
sim.compute_svd(sim.states[0], sim.temp)
print(state[2][0])
print(sim.temp.F.download()[0])
print(sim.temp.sig.download()[0])
# import torch
# print(torch.svd(torch.tensor(sim.temp.F.download()[0].reshape(-1, 3, 3), device='cuda:0')))

print(np.linalg.svd(sim.temp.F.download()[0].reshape(-1, 3, 3))[1])
"""

import tqdm
import time
for i in tqdm.trange(100000):
    # cu_env.renderer.render(cu_env.simulator, cu_env.simulator.states[i%1000])
    sim.temp.clear()
    sim.temp.clear_grad()
    temp = sim.temp
    cur = sim.states[i%1000]
    next = sim.states[(i+1)%1000]
    cur.clear_grad(sim.stream1)
    next.clear_grad(sim.stream1)

    sim.compute_svd(cur, temp) 
    sim.p2g(cur, temp, next)
    sim.grid_op(cur, temp, next)
    sim.g2p(cur, temp, next)

    sim.g2p_grad(cur, temp, next)
    sim.grid_op_grad(cur, temp, next)
    sim.p2g_grad(cur, temp, next)
    sim.compute_svd_grad(cur, temp)

    #sim.compute_grid_mass(i%1000, -1)