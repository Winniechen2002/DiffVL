# this function will update the policy per-step...
import numpy as np
import torch
import taichi as ti
from plb.engine.taichi_env import TaichiEnv
from torch.autograd import Function
import logging

from zmq import device


@ti.data_oriented
class DiffModel:
    def __init__(
        self,
        env: TaichiEnv,
        return_grid=(-1,), # -1 means all
        return_dist=True,
        return_svd=False # return the lambda, e.g., the eigenvalue of each deformation gradients.. in form of nx3x3
    ):
        self.env = env
        self.sim = env.simulator
        self.dim = self.sim.dim
        assert return_grid[0] == -1

        #self.substeps = self.sim.substeps
        self.primitives = self.sim.primitives
        self._forward_func = None

        self.return_grid = return_grid
        self.return_dist = return_dist
        self.return_svd = return_svd

    @property
    def substeps(self):
        return self.sim.substeps

    def zero_grad(self, return_grid=None, return_svd=None, max_steps=None, horizon=None):
        assert horizon is None or max_steps is None
        if horizon is not None:
            max_steps = (horizon + 1) * self.substeps
        if max_steps is None:
            max_steps = self.sim.max_steps
        # print(horizon, max_steps)

        assert max_steps <= self.sim.max_steps, f"{max_steps} {self.sim.max_steps}"
        #self.sim.x.grad.fill(0)
        #self.sim.v.grad.fill(0)
        #self.sim.F.grad.fill(0)
        #self.sim.C.grad.fill(0)
        self.sim.clear_grad(max_steps)

        for i in self.primitives:
            #if i.action_dim > 0:
            #    i.action_buffer.grad.fill(0)
            #i.position.grad.fill(0)
            #i.rotation.grad.fill(0)
            #i.v.grad.fill(0)
            #i.w.grad.fill(0)
            i.clear_grad(max_steps)

            assert i.action_dim <= 6
            #if i.action_dim == 7:
            #    raise NotImplementedError
            #    # i.gap.grad.fill(0)
            #    # i.gap_vel.grad.fill(0)

        if return_grid is not None:
            self.return_grid = return_grid
        if return_svd is not None:
            self.return_svd = return_svd or self.return_svd


    @ti.kernel
    def _get_obs(self, s: ti.int32, x: ti.ext_arr(), c: ti.ext_arr()):
        for i in range(self.sim.n_particles[None]):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.sim.x[s, i][j]
                x[i, j+self.dim] = self.sim.v[s, i][j]
        for idx, i in ti.static(enumerate(self.primitives)):
            if ti.static(i.action_dim>0):
                for j in ti.static(range(i.pos_dim)):
                    c[idx, j] = i.position[s][j]
                for j in ti.static(range(i.rotation_dim)):
                    c[idx, j+i.pos_dim] = i.rotation[s][j]

    def get_obs(self, s, device):
        f = s * self.sim.substeps
        x = torch.zeros(size=(self.sim.n_particles[None], self.dim * 2), device=device)
        c = torch.zeros(size=(len(self.primitives), self.primitives[0].state_dim), device=device)
        self._get_obs(f, x, c)

        if self.return_dist:
            dists = self.sim.get_dists(f, device=device)
            x = torch.cat((x, dists), 1)

        outputs = [x.clone(), c.clone()]

        if len(self.return_grid) > 0:
            outputs += [
                self.sim.compute_grid_mass(
                    f, i, device)
                for i in self.return_grid
            ]

        if self.return_svd:
            outputs.append(
                self.sim.compute_svd(f, device, None)
            )

        return tuple(outputs)


    def wrap_obs(self, obs):
        # assert len(obs) == 2
        output = {
            'pos': obs[0][:, :3],
            'vel': obs[0][:, 3:6],
            'tool': obs[1],
        }
        if self.return_dist:
            output['dist'] = obs[0][:, 6:]
        #assert len(self.output_grid) == 0
        obs = obs[2:]
        if len(self.return_grid) > 0:
            ng = len(self.return_grid)
            grid, obs = obs[:ng], obs[ng:]
            output['grid'] = {
                k: v for k, v in zip(self.return_grid, grid)
            }
        if self.return_svd:
            output['eigen'], obs = obs[0], obs[1:]
        assert len(obs) == 0
        return output


    @ti.kernel
    def _set_obs_grad(self, s:ti.int32, x:ti.ext_arr(), c:ti.ext_arr()):
        for i in range(self.sim.n_particles[None]):
            for j in ti.static(range(self.dim)):
                self.sim.x.grad[s, i][j] += x[i, j]
                self.sim.v.grad[s, i][j] += x[i, j+self.dim]
        for idx, i in ti.static(enumerate(self.primitives)):
            if i.action_dim > 0:
                for j in ti.static(range(i.pos_dim)):
                    i.position.grad[s][j] += c[idx, j]
                for j in ti.static(range(i.rotation_dim)):
                    i.rotation.grad[s][j] += c[idx, j+i.pos_dim]

    def set_obs_grad(self, s, obs_grad, manipulator_grad, *args):
        f = s * self.sim.substeps

        if self.return_svd:
            svd_grad, args = args[-1], args[:-1]
            self.sim.compute_svd(f, svd_grad.device, backward_grad=svd_grad)

        if len(self.return_grid) > 0:
            for idx, i in enumerate(self.return_grid):
                self.sim.compute_grid_mass(f, i, backward_grad=args[idx])

        if self.return_dist:
            start = -len(self.sim.dists)
            dist_grads = obs_grad[:, start:].clone().contiguous()
            #self.compute_min_dist(f)
            #self.torch2dist_grad(dist_grads)
            #self.compute_min_dist.grad(f)
            self.sim.get_dists(f, dist_grads)
            obs_grad = obs_grad[..., :start].clone()

        obs_grad = obs_grad.reshape(-1, self.dim * 2)
        manipulator_grad = manipulator_grad.reshape(len(self.primitives), self.primitives[0].state_dim)
        self._set_obs_grad(f, obs_grad, manipulator_grad)

    def forward_step(self, s, a):
        a = a.reshape(-1).clamp(-1, 1)
        for i in range(len(self.primitives)):
            if self.primitives[i].action_dim > 0:
                aa = a[self.primitives.action_dims[i]:self.primitives.action_dims[i+1]].clone()
                assert len(aa) == self.primitives[i].action_dim, f"{a.shape} {aa.shape}, {self.primitives[i].action_dim}"
                self.primitives[i].set_action(s, self.substeps, aa)

        for i in range(s*self.substeps, (s+1) * self.substeps):
            self.sim.substep(i)

    def backward_step(self, s):
        for i in range((s+1) * self.substeps-1, s*self.substeps-1, -1):
            self.sim.substep_grad(i)
        for i in self.primitives:
            if i.action_dim > 0:
                i.set_velocity.grad(s, self.substeps)

        grads = []
        for i in self.primitives:
            if i.action_dim > 0:
                grad = i.get_action_grad(s, 1)
                assert grad.shape[0] == 1
                grads.append(grad[0])
        return torch.tensor(np.concatenate(grads), device=self.device)

    @property
    def forward(self):
        if self._forward_func is None:
            class forward(Function):
                @staticmethod
                def forward(ctx, s, a, *past_obs):
                    ctx.save_for_backward(torch.tensor([s]), *[torch.zeros_like(i) for i in past_obs])
                    self.forward_step(s, a) # the model calculate one step forward
                    return self.get_obs(s+1, a.device) # get the observation at timestep s + 1

                @staticmethod
                def backward(ctx, *obs_grad):
                    # todo: use cur to check/force the backward orders..
                    tmp = ctx.saved_tensors
                    s = tmp[0].item()
                    self.set_obs_grad(s+1, *obs_grad) # add the gradient back into the tensors ...
                    actor_grad = self.backward_step(s)
                    return (None, actor_grad.reshape(-1)) + tmp[1:]
            self._forward_func = forward
        return self._forward_func.apply
