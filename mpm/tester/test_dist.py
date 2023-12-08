import pickle
from mpm.cuda_env import CudaEnv
import os

from mpm.tester.tester import generate_state, assert_state, test_p2g
import numpy as np

cu_env = CudaEnv(PRIMITIVES=[], SIMULATOR=dict(n_particles=20000))

with open('g2p_checkpoint.pkl', 'rb') as f:
    #state, state2, grid_m, grid_v_in, old_grid_v_out, grid_v_out2 = pickle.load(f)
    data = pickle.load(f)

state, state2, grid_v_out, out_x_grad, out_v_grad, out_C_grad, x_out_gt, v_out_gt, C_out_gt, inp_x_grad_gt, inp_v_grad_gt, grid_v_out_grad_gt = data



n = len(state[0])
sim = cu_env.simulator
sim.set_state(0, state)
sim.get_dists(0)