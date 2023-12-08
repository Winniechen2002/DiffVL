import numpy as np
import pickle
from ..cuda_env import CudaEnv

def generate_state(n=10000):

    x = np.random.random(size=(n, 3)) * 0.1 + np.array((0.3, 0.3, 0.3))
    v = np.random.random(size=(n, 3))

    F = np.random.normal(size=(n, 3, 3))
    C = np.random.normal(size=(n, 3, 3))
    p1 = np.zeros(7)
    p2 = np.zeros(7)
    p1[4] = 1
    p2[4] = 1

    p1[:3] = np.random.random((3,))
    p2[:3] = np.random.random((3,))

    w1 = np.random.normal(size=4)
    p1[3:] = w1/np.linalg.norm(w1)
    w2 = np.random.normal(size=4)
    p2[3:] = w2/np.linalg.norm(w2)
    return x, v, (F[:, :, None] * F[:, None,:] + np.eye(3)*0.01).sum(-1), C, p1, p2

def assert_state(a, b):
    for i, j in zip(a, b):
        assert np.allclose(i, j)

def test_svd_grad(
    cu_env: CudaEnv,
    plb_env,
    state,
    seed=0,
):
    sim = cu_env.simulator

    sim.set_state(0, state)
    print(sim.n_particles, type(sim.n_particles))
    assert_state(sim.get_state(0), state)
    sim.compute_svd(sim.states[0], sim.temp) # 

    np.random.seed(0)

    F_tmp_grad = np.random.normal(size=(sim.n_particles, 3, 3))
    U_grad = np.random.normal(size=(sim.n_particles, 3, 3))
    S_grad = np.random.normal(size=(sim.n_particles, 3))
    V_grad = np.random.normal(size=(sim.n_particles, 3, 3))

    sim.states[0].clear_grad()
    sim.temp.clear_grad()
    sim.temp.F_grad.upload(F_tmp_grad.reshape(-1, 9))
    sim.temp.V_grad.upload(V_grad.reshape(-1, 9))
    sim.temp.sig_grad.upload(S_grad.reshape(-1, 3))
    sim.temp.U_grad.upload(U_grad.reshape(-1, 9))
    sim.compute_svd_grad(sim.states[0], sim.temp)

    F1_grad = sim.states[0].F_grad.download().reshape(-1, 3, 3)
    C1_grad = sim.states[0].C_grad.download().reshape(-1, 3, 3)

    F_tmp = sim.temp.F.download().reshape(-1, 3, 3)
    U = sim.temp.U.download().reshape(-1, 3, 3)
    sig = sim.temp.sig.download().reshape(-1, 3)
    V = sim.temp.V.download().reshape(-1, 3, 3)



    plb_env.simulator.clear_SVD_grad()
    plb_env.simulator.set_state(0, state)
    plb_env.simulator.clear_grad(10)
    plb_env.simulator.compute_F_tmp(0)
    plb_env.simulator.svd()

    plb_env.simulator.F_tmp.grad.from_numpy(F_tmp_grad)
    plb_env.simulator.U.grad.from_numpy(U_grad)
    sig_grad_2 = np.zeros((sim.n_particles, 3, 3))
    sig_grad_2[:, [0, 1,2], [0, 1, 2]] = S_grad
    plb_env.simulator.sig.grad.from_numpy(sig_grad_2)
    plb_env.simulator.V.grad.from_numpy(V_grad)

    plb_env.simulator.svd_grad()
    plb_env.simulator.compute_F_tmp.grad(0)

    F_tmp2 = plb_env.simulator.F_tmp.to_numpy()
    U2 = plb_env.simulator.U.to_numpy()
    sig2 = plb_env.simulator.sig.to_numpy()[:, [0, 1, 2], [0, 1, 2]]
    V2 = plb_env.simulator.V.to_numpy()

    assert np.allclose(F_tmp, F_tmp2)
    #print('max diff', np.abs(sig2-sig).max())
    assert np.allclose(sig2, sig, atol=1e-6, rtol=1e-1), f"{np.abs(U2 -U).argmax()}"

    same_u_v = np.abs(U2-U).max(axis=(1, 2)) < 1e-6

    F_tmp2_grad_end = plb_env.simulator.F_tmp.grad.to_numpy()[same_u_v]
    F_tmp_grad_end = sim.temp.F_grad.download().reshape(-1, 3, 3)[same_u_v]

    F2_grad = plb_env.simulator.F.grad.to_numpy()[0][same_u_v]
    C2_grad = plb_env.simulator.C.grad.to_numpy()[0][same_u_v]
    F1_grad = F1_grad[same_u_v]
    C1_grad = C1_grad[same_u_v]

    #print(F_tmp_grad[0])

    #print(F_tmp_grad_end[0])
    #print(F_tmp2_grad_end[0])
    assert abs(cu_env.simulator.dt - plb_env.simulator.dt[None]) < 1e-8

    diff = F_tmp_grad_end - F_tmp2_grad_end
    x = np.abs(diff).argmax()//9
    print(F_tmp_grad_end[x], F_tmp2_grad_end[x])
    assert diff.max() < 0.1, f"{diff.max()}"

    print(F_tmp_grad_end[0] @ state[2][0] * cu_env.simulator.dt)

    print(F1_grad[0], F2_grad[0])
    print(np.abs(F1_grad - F2_grad).max())
    print(np.abs(C1_grad - C2_grad).max())

    assert np.abs(F1_grad - F2_grad).max() < 0.1
    assert np.abs(C1_grad - C2_grad).max() < 2e-6

    
def test_p2g(
    cu_env: CudaEnv,
    plb_env,
    state,
    seed=0,
):

    sim = cu_env.simulator
    sim.particle_mass.upload(np.ones(sim.n_particles))
    sim.set_state(0, state)
    assert_state(sim.get_state(0), state)
    sim.compute_svd(sim.states[0], sim.temp) # 

    sim.temp.clear()
    sim.p2g(sim.states[0], sim.temp, sim.states[1])

    grid_m = sim.temp.grid_m.download().reshape(64, 64, 64)
    grid_v_in = sim.temp.grid_v_in.download().reshape(64, 64, 64, 3)
    next_F = sim.states[1].F.download().reshape(-1, 3, 3)
    F_tmp = sim.temp.F.download()

    from diffrl.env import DifferentiablePhysicsEnv
    plb_env: DifferentiablePhysicsEnv = plb_env
    plb_env.simulator.p_mass.fill(1.)

    plb_env.simulator.yield_stress.fill(sim.particle_mu_lam_yield.download()[0][-1])
    plb_env.simulator.set_state(0, state)
    plb_env.simulator.clear_grad(10)
    plb_env.simulator.clear_grid()
    plb_env.simulator.clear_SVD_grad()
    plb_env.simulator.compute_F_tmp(0)
    plb_env.simulator.svd()
    plb_env.simulator.p2g(0)

    grid_m_grad = (np.random.random(size=(64, 64, 64)) * 2 - 0.3)
    grid_v_in_grad = ((np.random.random(size=(64, 64, 64, 3)) * 2 - 0.3))
    next_F_grad = np.random.random(size=(len(state[0]), 3, 3)) * 2 -1


    plb_sim = plb_env.simulator

    plb_sim.x.grad.fill(0.)
    plb_sim.v.grad.fill(0.)
    plb_sim.F_tmp.grad.fill(0.)
    plb_sim.C.grad.fill(0.)
    plb_sim.U.grad.fill(0.)
    plb_sim.sig.grad.fill(0.)
    plb_sim.V.grad.fill(0.)

    plb_sim.grid_m.grad.from_numpy(grid_m_grad)
    plb_sim.grid_v_in.grad.from_numpy(grid_v_in_grad)
    F_grad = plb_sim.F.grad.to_numpy() * 0
    F_grad[1] = next_F_grad
    plb_sim.F.grad.from_numpy(F_grad)
    plb_sim.p2g.grad(0)

    input_grads = {
        'grid_m_grad': grid_m_grad,
        'grid_v_in_grad': grid_v_in_grad,
        'next_F_grad': next_F_grad,
    }

    output_grads = {
        'x_grad': plb_sim.x.grad.to_numpy()[0],
        'v_grad': plb_sim.v.grad.to_numpy()[0],
        'F_tmp_grad': plb_sim.F_tmp.grad.to_numpy(),
        'C_grad': plb_sim.C.grad.to_numpy()[0],
        'U_grad': plb_sim.U.grad.to_numpy(),
        'V_grad': plb_sim.V.grad.to_numpy(),
        'sig_grad': plb_sim.sig.grad.to_numpy(),
        'sig': plb_sim.sig.to_numpy(),
    }

    grid_m_2 = plb_sim.grid_m.to_numpy()
    grid_v_in_2 = plb_sim.grid_v_in.to_numpy()
    next_F_2 = plb_sim.F.to_numpy()[1]
    F_tmp_2 = plb_sim.F_tmp.to_numpy()

    with open('p2g_grad.pkl', 'wb') as f:
        pickle.dump([state, input_grads, output_grads, [grid_m_2, grid_v_in_2, next_F_2, F_tmp_2]], f)




    max_id =  np.abs(next_F - next_F_2).argmax()//9

    print(sim.temp.sig.download()[max_id])
    print(plb_env.simulator.sig.to_numpy()[max_id][[0,1,2], [0,1,2]])

    print('np', np.linalg.svd(F_tmp_2[max_id].reshape(-1, 3, 3))[1])

    print(plb_sim.mu[0], plb_sim.lam[0], plb_sim.yield_stress[0])
    print(sim.particle_mu_lam_yield.download()[0])
    max_diff = np.abs(next_F[max_id]-next_F_2[max_id]).max()
    assert max_diff < 0.01, f"{max_diff}"

    print(plb_sim.dx)
    print('mass', sim.particle_mass.download()[0])
    print('mass', plb_sim.p_mass[0])


    print('max grid m abs', np.abs(grid_m).max(), np.abs(grid_m_2).max())
    print('max grid m gap', np.abs(grid_m - grid_m_2).max())
    print('max grid v abs', np.abs(grid_v_in).max(), np.abs(grid_v_in_2).max())
    print('max grid v gap', np.abs(grid_v_in - grid_v_in_2).max())
    assert np.allclose(grid_v_in, grid_v_in_2, atol=1e-4, rtol=1e-2)



def test_grid_op(
    cu_env: CudaEnv,
    plb_env,
    state,
    state2,
    seed=0,
):
    np.random.seed(0)

    grid_m = np.float32(np.random.random(size=(64, 64, 64)))
    grid_v_in = np.float32(np.random.normal(size=(64, 64, 64, 3)))
    grid_v_out_grad = np.float32(np.random.normal(size=(64, 64, 64, 3)))

    if False:
        mask = np.zeros((64, 64, 64))
        mask[31, 39, 52] = 1.
        grid_m = grid_m * mask



    sim = cu_env.simulator
    sim.set_state(0, state)
    sim.set_state(1, state2)
    sim.dt=float(1)

    sim.temp.grid_m.upload(grid_m.reshape(-1))
    sim.temp.grid_v_in.upload(grid_v_in.reshape(-1, 3))


    sim.grid_op(sim.states[0], sim.temp, sim.states[1])

    grid_v_out = sim.temp.grid_v_out.download().reshape(64, 64, 64, 3)


    from diffrl.env import DifferentiablePhysicsEnv
    plb_env: DifferentiablePhysicsEnv = plb_env
    plb_sim = plb_env.simulator
    plb_sim.dt[None] = 1.

    for i in plb_sim.primitives.primitives:
        i.softness[None] = 666.

    plb_sim.set_state(0, state)
    plb_sim.set_state(1, state2)
    plb_sim.clear_grid()
    plb_sim.grid_m.from_numpy(grid_m)
    plb_sim.grid_v_in.from_numpy(grid_v_in)
    plb_sim.grid_op(0)
    plb_sim.grid_v_out.grad.from_numpy(grid_v_out_grad)
    for i in plb_sim.primitives.primitives:
        i.clear_grad(10)
    plb_sim.grid_op.grad(0)

    pos_grad = []
    rot_grad = []
    next_pos_grad = []
    next_rot_grad = []
    for i in plb_sim.primitives.primitives:
        pos_grad.append(i.position.grad.to_numpy()[0])
        next_pos_grad.append(i.position.grad.to_numpy()[1])
        rot_grad.append(i.rotation.grad.to_numpy()[0])
        next_rot_grad.append(i.rotation.grad.to_numpy()[1])
    

    grid_v_out2 = plb_sim.grid_v_out.to_numpy()

    print(grid_v_out.max())
    print(grid_v_out2.max())

    print(np.abs(grid_v_out - grid_v_out2).max())

    print(sim.body_type_mu_softness_round.download())
    print(sim.body_args.download())

    
    print('dt', float(sim.dt), 'and', float(plb_sim.dt[None]))
    for i in plb_sim.primitives.primitives:
        print('mu', i.friction[None], 'soft', i.softness[None], 'round', i.round[None])
        print('size', i.size[None])
        print('stiff', i.stiffness[None])


    inputs = [state, state2, grid_m, grid_v_in, grid_v_out_grad]
    outputs = [plb_sim.grid_m.grad.to_numpy(), plb_sim.grid_v_in.grad.to_numpy(), plb_sim.grid_v_out.to_numpy(), np.array(pos_grad), np.array(rot_grad), np.array(next_pos_grad), np.array(next_rot_grad)]
    
    #assert np.allclose(grid_v_out, grid_v_out2), f"{np.abs(grid_v_out - grid_v_out2).max()}"
    with open('test_grid_op_checkpoint_v2.pkl', 'wb') as f:
        #pickle.dump([state, state2, grid_m, grid_v_in, grid_v_out, grid_v_out2], f)
        pickle.dump([inputs, outputs], f)


def test_g2p(
    cu_env: CudaEnv,
    plb_env,
    state,
    state2,
    seed=0,
):
    sim = cu_env.simulator
    sim.set_state(0, state)
    sim.set_state(1, state2)


    n = len(state[0])
    np.random.seed(0)
    grid_v_out = np.float32(np.random.normal(size=(64, 64, 64, 3)))
    out_x_grad = np.float32(np.random.normal(size=(n, 3)))
    out_v_grad = np.float32(np.random.normal(size=(n, 3))) 
    out_C_grad = np.float32(np.random.normal(size=(n, 3, 3)))


    sim.temp.grid_v_out.upload(grid_v_out.reshape(-1, 3))
    sim.temp.clear_grad()
    sim.states[0].clear_grad()
    sim.states[1].clear_grad()

    sim.g2p(sim.states[0], sim.temp, sim.states[1])

    sim.states[1].x_grad.upload(out_x_grad)
    sim.states[1].v_grad.upload(out_v_grad)
    sim.states[1].C_grad.upload(out_C_grad.reshape(n, 9))

    sim.g2p_grad(sim.states[0], sim.temp, sim.states[1])

    #x_grad = sim.states[0].x_grad.download()
    #v_grad = sim.states[0].v_grad.download()
    #C_grad = sim.states[0].v_grad.download()
    from diffrl.env import DifferentiablePhysicsEnv
    plb_env: DifferentiablePhysicsEnv = plb_env

    plb_sim = plb_env.simulator
    plb_sim.set_state(0, state)
    plb_sim.set_state(1, state)

    plb_sim.clear_grad(10)
    plb_sim.clear_grid()
    plb_sim.clear_SVD_grad()

    plb_sim.grid_v_out.from_numpy(grid_v_out)
    plb_sim.g2p(0)

    x_grad = plb_sim.x.grad.to_numpy() * 0
    x_grad[1] = out_x_grad
    plb_sim.x.grad.from_numpy(x_grad)

    v_grad = plb_sim.v.grad.to_numpy() * 0
    v_grad[1] = out_v_grad
    plb_sim.v.grad.from_numpy(v_grad)

    C_grad = plb_sim.C.grad.to_numpy() * 0
    C_grad[1] = out_C_grad
    plb_sim.C.grad.from_numpy(C_grad)

    print(out_v_grad.sum(), out_C_grad.sum())
    

    plb_sim.g2p.grad(0)

    x_out = plb_sim.x.to_numpy()[1]
    v_out = plb_sim.v.to_numpy()[1]
    C_out = plb_sim.C.to_numpy()[1]

    inp_x_grad = plb_sim.x.grad.to_numpy()[0]
    inp_v_grad = plb_sim.v.grad.to_numpy()[0]
    grid_v_out_grad = plb_sim.grid_v_out.grad.to_numpy()

    data = [state, state2, grid_v_out, out_x_grad, out_v_grad, out_C_grad, x_out, v_out, C_out, inp_x_grad, inp_v_grad, grid_v_out_grad]

    import pickle
    with open('g2p_checkpoint.pkl', 'wb') as f:
        pickle.dump(data, f)