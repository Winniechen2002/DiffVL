import pickle
from mpm.cuda_env import CudaEnv
import os

from mpm.tester import generate_state, assert_state, test_p2g
import numpy as np

cu_env = CudaEnv(PRIMITIVES=[], SIMULATOR=dict(n_particles=20000))

with open('g2p_checkpoint.pkl', 'rb') as f:
    #state, state2, grid_m, grid_v_in, old_grid_v_out, grid_v_out2 = pickle.load(f)
    data = pickle.load(f)

state, state2, grid_v_out, out_x_grad, out_v_grad, out_C_grad, x_out_gt, v_out_gt, C_out_gt, inp_x_grad_gt, inp_v_grad_gt, grid_v_out_grad_gt = data



n = len(state[0])
sim = cu_env.simulator
sim.set_state(0, state)
sim.set_state(1, state2)

sim.temp.grid_v_out.upload(grid_v_out.reshape(-1, 3))
sim.temp.clear_grad()
sim.states[0].clear_grad()
sim.states[1].clear_grad()

sim.g2p(sim.states[0], sim.temp, sim.states[1])
x_out = sim.states[1].x.download().reshape(n, 3)
v_out = sim.states[1].v.download().reshape(n, 3)
C_out = sim.states[1].C.download().reshape(n, 3, 3)


sim.states[1].x_grad.upload(out_x_grad)
sim.states[1].v_grad.upload(out_v_grad)
sim.states[1].C_grad.upload(out_C_grad.reshape(n, 9))

sim.g2p_grad(sim.states[0], sim.temp, sim.states[1])



x_in = sim.states[0].x.download().reshape(-1, 3)

print(v_out.sum())
p = np.abs(v_out - v_out_gt).argmax()//3

pid = np.argmax(np.abs(x_out-x_out_gt))//3
print(pid, x_in[pid], x_out[pid], x_out_gt[pid])
print(x_out.max(), x_out_gt.max())
assert np.allclose(x_out, x_out_gt, rtol=1e-6)
assert np.allclose(v_out, v_out_gt, atol=1e-6, rtol=1e-6)
assert np.allclose(C_out, C_out_gt, atol=1e-4, rtol=1e-6)

inp_x_grad = sim.states[0].x_grad.download().reshape(n, 3)
grid_v_out_grad = sim.temp.grid_v_out_grad.download().reshape(64, 64, 64, 3)
pid = np.abs(grid_v_out_grad - grid_v_out_grad_gt).argmax()//3
x = pid
z = x % 64
x//=64
y = x % 64
x//=64
print(x, y, z, grid_v_out_grad[x, y, z], grid_v_out_grad_gt[x, y, z])
assert np.allclose(grid_v_out_grad, grid_v_out_grad_gt, atol=1e-2, rtol=1e-6)

pid = np.argmax(np.abs(inp_x_grad-inp_x_grad_gt))//3
print(inp_x_grad[pid], inp_x_grad_gt[pid], np.abs(inp_x_grad-inp_x_grad_gt).max())
assert np.allclose(inp_x_grad, inp_x_grad_gt, atol=2e-2, rtol=1e-6)