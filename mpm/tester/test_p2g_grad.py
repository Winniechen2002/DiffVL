import numpy as np
import pickle
with open('p2g_grad.pkl', 'rb') as f:
    state, input_grads, output_grads, [grid_m_2, grid_v_in_2, next_F_2, F_tmp_2] = pickle.load(f)
print(input_grads.keys())
print(output_grads.keys())

from mpm.cuda_env import CudaEnv
import os
cu_env = CudaEnv(PRIMITIVES=[], SIMULATOR=dict(yield_stress=10000.))
sim = cu_env.simulator
sim.particle_mass.upload(np.ones(sim.n_particles))

sim.temp.clear()
sim.temp.clear_grad()
sim.states[0].clear_grad(sim.stream0)

sim.set_state(0, state)
sim.compute_svd(sim.states[0], sim.temp) # 
sim.p2g(sim.states[0], sim.temp, sim.states[1])

print(input_grads.keys())
sim.temp.grid_m_grad.upload(input_grads['grid_m_grad'].reshape(-1))
sim.temp.grid_v_in_grad.upload(input_grads['grid_v_in_grad'].reshape(-1, 3))
sim.states[1].F_grad.upload(input_grads['next_F_grad'].reshape(-1, 3 * 3))
sim.p2g_grad(sim.states[0], sim.temp, sim.states[1])

v_grad = sim.states[0].v_grad.download().reshape(-1, 3)
C_grad = sim.states[0].C_grad.download().reshape(-1, 3, 3)
x_grad = sim.states[0].x_grad.download().reshape(-1, 3)



grid_m = sim.temp.grid_m.download().reshape(64, 64, 64)
grid_v_in = sim.temp.grid_v_in.download().reshape(64, 64, 64, 3)
next_F = sim.states[1].F.download().reshape(-1, 3, 3)
F_tmp = sim.temp.F.download()

pp = np.abs(grid_v_in - grid_v_in_2).argmax()//3
x, y, z = pp//64//64, (pp//64) %64, pp%64
print('v', np.abs(grid_v_in - grid_v_in_2).max(), grid_v_in[x,y,z], grid_v_in_2[x,y,z])
assert np.allclose(grid_v_in, grid_v_in_2, atol=2e-4, rtol=1e-6)
assert np.allclose(v_grad, output_grads['v_grad'])
print(C_grad.max(axis=0))
print(output_grads['C_grad'].max(axis=0))
assert np.allclose(C_grad, output_grads['C_grad'])

print('x_grad', x_grad.max(axis=0), x_grad.sum())
assert np.allclose(x_grad, output_grads['x_grad'], atol=1e-1, rtol=1e-8)



grad_U = sim.temp.U_grad.download().reshape(-1, 3, 3)
grad_V = sim.temp.V_grad.download().reshape(-1, 3, 3)
grad_sig = sim.temp.sig_grad.download().reshape(-1, 3)
grad_F = sim.temp.F_grad.download().reshape(-1, 3, 3)


diffU =np.abs(grad_U - output_grads['U_grad'])
pp = diffU.argmax()//9
print(diffU[pp].max())
print(grad_U[pp],'\n', output_grads['U_grad'][pp])
print(np.abs(np.abs(grad_U) - np.abs(output_grads['U_grad'])).max())
print("NOTE that the sign of U and V is undertermined!")
assert np.allclose(np.abs(grad_U), np.abs(output_grads['U_grad']), atol=2e-2, rtol=1e-6)

diffV =np.abs(np.abs(grad_V) - np.abs(output_grads['V_grad']))
pp = diffV.argmax()//9
print(diffV[pp].max())
print(grad_V[pp], output_grads['V_grad'][pp])
assert np.allclose(np.abs(grad_V), np.abs(output_grads['V_grad']), atol=1e-2, rtol=1e-6)


diffF =np.abs(grad_F - output_grads['F_tmp_grad'])
pp = diffF.argmax()//9
print("grad F")
print(diffF.max())
print(grad_F[pp], '\n', output_grads['F_tmp_grad'][pp])
assert np.allclose(grad_F, output_grads['F_tmp_grad'], atol=1e-4, rtol=1e-6)

sig_grad_gt = output_grads['sig_grad'][:, [0,1,2], [0,1,2]]
diff_sig =np.abs(grad_sig - sig_grad_gt)
pp = diff_sig.argmax()//3
print(diff_sig.max(), pp)
print(grad_sig[pp], sig_grad_gt[pp])
print(sim.temp.sig.download()[pp])
print(output_grads['sig'][pp, [0,1,2],[0,1,2]])
assert np.allclose(grad_sig, sig_grad_gt, atol=2e-2, rtol=1e-6)