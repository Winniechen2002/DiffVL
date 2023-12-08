# import sapien.core as sapien
import numpy as np


def dump_grad(sim, index):
    n = sim.n_particles[None]
    x = sim.x.grad.to_numpy()[index, :n]
    v = sim.v.grad.to_numpy()[index, :n]
    F = sim.F.grad.to_numpy()[index, :n]
    C = sim.C.grad.to_numpy()[index, :n]
    out = [x, v, F, C]
    for i in sim.primitives:
        out.append(np.r_[i.position.grad.to_numpy()[index], i.rotation.grad.to_numpy()[index]])
    return out


def dump_taichi_grad(sim, index):
    state = sim.get_state(index)
    state_grad = dump_grad(sim, index)
    return {'state': state, 'grad': state_grad}


def dump_substeps(sim, index):
    a = dump_taichi_grad(sim, index)
    b = dump_taichi_grad(sim, index + 1)
    # sim.substep_grad(index)

    n = sim.n_particles[None]

    self = sim
    s = index
    self.clear_grid()
    self.clear_SVD_grad()  # clear the svd grid

    self.compute_F_tmp(s)  # we need to compute it for calculating the svd decomposition
    self.svd()
    self.p2g(s)
    self.grid_op(s)

    self.g2p.grad(s)

    self.grid_op.grad(s)
    self.p2g.grad(s)
    for i in range(self.n_primitive - 1, -1, -1):
        self.primitives[i].forward_kinematics.grad(s)
    inter_F = self.F_tmp.grad.to_numpy()[:n]
    self.svd_grad()
    self.compute_F_tmp.grad(s)

    return [a, b,
            [
                sim.grid_v_in.to_numpy(),
                sim.grid_v_in.grad.to_numpy(),
                sim.grid_v_out.to_numpy(),
                sim.grid_v_out.grad.to_numpy(),
                sim.grid_m.to_numpy(),
                sim.grid_m.grad.to_numpy(),
                inter_F,
                [sim.U.grad.to_numpy()[:n], sim.V.grad.to_numpy()[:n], sim.sig.grad.to_numpy()[:n],
                 sim.F_tmp.grad.to_numpy()[:n]],
                [sim.U.to_numpy()[:n], sim.V.to_numpy()[:n], sim.sig.to_numpy()[:n], sim.F_tmp.to_numpy()[:n]],
            ]]


def set_mpm_grad(sim, index, x):
    # sim.set_state(index, x['state'])
    sim.set_state(index, x['state'])

    s = sim.states[index]
    grad = x['grad']
    s.x_grad.upload(grad[0])
    s.v_grad.upload(grad[1])
    s.F_grad.upload(grad[2].reshape(-1, 9))
    s.C_grad.upload(grad[3].reshape(-1, 9))

    grad = np.array(grad[4:])
    s.body_pos_grad.upload(grad[:, :3])
    s.body_rot_grad.upload(grad[:, 3:])


def compare_grid(A, B):
    A = A.reshape(B.shape)
    v = np.abs(A - B).argmax()
    if len(B.shape) == 4:
        v = v // 3
    z = v % 64;
    v //= 64
    y = v % 64;
    v //= 64
    x = v % 64;
    v //= 64
    aa = A[x, y, z]
    bb = B[x, y, z]
    print(x, y, z, aa, bb, np.abs(aa - bb).max())


def compare_p(A, B, message=''):
    A = A.reshape(B.shape)
    v = np.abs(A - B).argmax()
    v = v // np.prod(A.shape[1:])
    aa = A[v]
    bb = B[v]
    print(message, v, aa, bb, np.abs(aa - bb).max())


def backward_svd(gu, gsigma, gv, u, sig, v):
    # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
    if False:
        gu = np.float64(gu)
        gsigma = np.float64(gsigma)
        gv = np.float64(gv)

        u = np.float64(u)
        sig = np.float64(sig)
        v = np.float64(v)

    vt = v.transpose()
    ut = u.transpose()
    sigma_term = u @ gsigma @ vt

    def clamp(a):
        # remember that we don't support if return in taichi
        # stop the gradient ...
        if a >= 0:
            a = max(a, 1e-6)
        else:
            a = min(a, -1e-6)
        return a

    # s = ti.Vector.zero(self.dtype, self.dim)
    s = np.array([sig[0, 0], sig[1, 1], sig[2, 2]], dtype=u.dtype) ** 2
    F = np.zeros((3, 3), dtype=u.dtype)
    for i in range(3):
        for j in range(3):
            if i == j:
                F[i, j] = 0
            else:
                F[i, j] = 1. / clamp(s[j] - s[i])
    u_term = u @ ((F * (ut @ gu - gu.transpose() @ u)) @ sig) @ vt
    v_term = u @ (sig @ ((F * (vt @ gv - gv.transpose() @ v)) @ vt))
    return u_term + v_term + sigma_term


def rigid_body_motion(states, actions):
    import torch
    # state: (B, 7)
    # action: (T, B, 6)
    T, B = actions.shape[:2]
    pos, q = states

    pos = pos[None, :].expand(T, -1, -1).reshape(-1, 3)
    q = q[None, :].expand(T, -1, -1).reshape(-1, 4)
    actions = actions.reshape(-1, 6)
    rot = actions[:, 3:]
    # better if we can move this part to C++ code
    w = torch.sqrt((rot * rot).sum(axis=-1, keepdims=True) + 1e-16)
    quat = torch.cat((torch.cos(w / 2), (rot / torch.clamp(w, 1e-7, 1e9)) * torch.sin(w / 2)), 1)

    next_pos = pos + actions[:, :3]
    # terms = q.outer_product(quat) # we should use w x q
    terms = q[:, :, None] * quat[:, None, :]
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    out = torch.stack([w, x, y, z], 1)
    next_rot = out/torch.linalg.norm(out, dim=-1, keepdims=True)
    return next_pos.reshape(T, B, 3), next_rot.reshape(T, B, 4)


def rigid_body_motion_hand(state, actions, T):
    """
    Args:
        state: (n_hands, 4, 4)
        action: (n_hands, 6,)

    Returns:
        state: (T, n_hands, 4, 4)
    """
    import torch
    from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion

    ####################################
    ###### Tile state and actions ######
    ####################################

    state = state[None, :].expand(T, -1, -1, -1).clone()
    actions = actions[None, :].expand(T, -1, -1).clone() * (
            torch.arange(T, device=actions.device)[:, None, None] + 1) / T

    #########################
    ###### translation ######
    #########################
    state[..., :3, 3] += actions[..., :3]

    ######################
    ###### rotation ######
    ######################
    q = matrix_to_quaternion(state[..., :3, :3])  # T, n_hands, 4
    rot = actions[..., 3:]  # T, n_hands, 3

    w = torch.sqrt((rot * rot).sum(axis=-1, keepdims=True) + 1e-16)  # T, n_hands
    quat = torch.cat((torch.cos(w / 2), (rot / torch.clamp(w, 1e-7, 1e9)) * torch.sin(w / 2)), -1)  # T, n_hands, 4
    # terms = q.outer_product(quat) # we should use w x q
    terms = q[..., None] * quat[..., None, :]
    w = terms[..., 0, 0] - terms[..., 1, 1] - terms[..., 2, 2] - terms[..., 3, 3]
    x = terms[..., 0, 1] + terms[..., 1, 0] - terms[..., 2, 3] + terms[..., 3, 2]
    y = terms[..., 0, 2] + terms[..., 1, 3] + terms[..., 2, 0] - terms[..., 3, 1]
    z = terms[..., 0, 3] - terms[..., 1, 2] + terms[..., 2, 1] + terms[..., 3, 0]
    out = torch.stack([w, x, y, z], -1)
    # print("out1", out[0])
    out = out / torch.linalg.norm(out, dim=-1, keepdims=True)
    # print("out2", out[0])
    state[..., :3, :3] = quaternion_to_matrix(out)

    return state
