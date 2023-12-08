import pickle
from mpm.cuda_env import CudaEnv
import os

from mpm.tester.tester import generate_state, assert_state, test_p2g
import numpy as np

cu_env = CudaEnv(SIMULATOR=dict(n_particles=20000, dt=1., ground_friction=10.))

with open('test_grid_op_checkpoint_v2.pkl', 'rb') as f:
    #state, state2, grid_m, grid_v_in, old_grid_v_out, grid_v_out2 = pickle.load(f)
    inputs, outputs = pickle.load(f)

state, state2, grid_m, grid_v_in, grid_v_out_grad = inputs
grid_m_grad_gt, grid_v_in_grad_gt, grid_v_out2, pos_grad_gt, rot_grad_gt, next_pos_grad_gt, next_rot_grad_gt = outputs




sim = cu_env.simulator
print(state[-1])
sim.set_state(0, state)
sim.set_state(1, state2)


sim.temp.grid_m.upload(grid_m.reshape(-1))
sim.temp.grid_v_in.upload(grid_v_in.reshape(-1, 3))


sim.grid_op(sim.states[0], sim.temp, sim.states[1])

sim.temp.clear_grad()
sim.states[0].clear_grad(sim.stream0)
sim.states[1].clear_grad(sim.stream0)
sim.temp.grid_v_out_grad.upload(grid_v_out_grad.reshape(-1,3))

grid_v_out = sim.temp.grid_v_out.download().reshape(64, 64, 64, 3)

sim.grid_op_grad(sim.states[0], sim.temp, sim.states[1])


pid =(np.linalg.norm(grid_v_out - grid_v_out2, axis=-1)).argmax()
x = pid
z = x % 64
x//=64
y = x % 64
x//=64
print(grid_v_out.max(axis=(0,1,2)), grid_v_out2.max(axis=(0, 1, 2)))
print(pid, x, y, z, grid_v_in[x, y, z])
print(pid, x, y, z, grid_v_out[x, y, z], grid_v_out2[x, y, z])
assert np.allclose(grid_v_out, grid_v_out2, atol=3, rtol=1e-6), f"{np.abs(grid_v_out - grid_v_out2).max()}"


grid_m_grad = sim.temp.grid_m_grad.download().reshape(64, 64, 64)
grid_v_in_grad = sim.temp.grid_v_in_grad.download().reshape(64, 64, 64, 3)

print(np.abs(grid_v_in_grad - grid_v_in_grad_gt).max())

pid =(np.linalg.norm(grid_v_in_grad - grid_v_in_grad_gt, axis=-1)).argmax()
x = pid
z = x % 64
x//=64
y = x % 64
x//=64
print(pid, x, y, z, grid_v_in_grad[x, y, z], grid_v_in_grad_gt[x, y, z])
print(grid_v_out[x, y, z], grid_v_out2[x, y, z])
assert np.allclose(grid_v_in_grad, grid_v_in_grad_gt, atol=1000,rtol=1e-8)

print(grid_m_grad.max(), grid_m_grad_gt.max())
print(np.abs(grid_m_grad - grid_m_grad_gt).max())
assert np.allclose(grid_m_grad, grid_m_grad_gt, atol=1000, rtol=1e-8)


next_pos_grad = sim.states[1].body_pos_grad.download()
next_rot_grad = sim.states[1].body_rot_grad.download()

print("+"*100)
print(next_pos_grad)
print(next_pos_grad_gt)


print("+"*100)
print(next_rot_grad)
print(next_rot_grad_gt)


pos_grad = sim.states[0].body_pos_grad.download()
rot_grad = sim.states[0].body_rot_grad.download()
print("+"*100)
print(pos_grad)
print(pos_grad_gt)

print("+"*100)
print(rot_grad)
print(rot_grad_gt)