import numpy as np
import torch
# from .cuda_env import CudaEnv
from .simulator import MPMSimulator
from torch.autograd import Function


class DiffModel:
    def __init__(self, env, return_grid=(-1,), return_svd=False):
        self.sim: MPMSimulator = env.simulator
        self.dim = 3
        if len(return_grid) > 0:
            assert return_grid[0] == -1
        self.primitives = [i for i in range(self.sim.n_bodies)]
        self._forward_func = None
        self._set_pose_func = None
        self.return_grid = return_grid
        self.return_svd = return_svd

        self.get_diff_forward()

    @property
    def substeps(self):
        return self.sim.substeps


    def zero_grad(self, return_grid=None, return_svd=None, **kwargs):
        self.device = 'cuda:0'
        self.sim.states[0].clear_grad(self.sim.stream1) # clear grad for states 0
        if return_grid is not None:
            self.return_grid = return_grid
        if return_svd is not None:
            self.return_svd = return_svd or self.return_svd


    def wrap_obs(self, obs):
        output = {
            'pos': obs[0][:, :3],
            'vel': obs[0][:, 3:6],
            'tool': obs[1],
        }
        output['dist'] = obs[0][:, 6:]

        obs = obs[2:]
        if len(self.return_grid) > 0:
            ng = len(self.return_grid)
            grid, obs = obs[:ng], obs[ng:]
            output['grid'] = {
                k: v for k, v in zip(self.return_grid, grid)
            }

        assert len(obs) == 0
        return output


    def get_obs(self, s, device):
        f = s * self.sim.substeps
        x = self.sim.get_x(f, device)
        v = self.sim.get_v(f, device)
        c = torch.stack(self.sim.get_tool_state(f, device))
        dists = self.sim.get_dists(f, device=device)

        x = torch.cat((x, v, dists), 1)

        outputs = [x, c]

        if len(self.return_grid) > 0:
            outputs += [
                self.sim.compute_grid_mass(
                    f, i, device=device)
                for i in self.return_grid
            ]

        return tuple(outputs)

    def set_obs_grad(self, s, particle_grad, tool_grad, *args):
        f = s * self.sim.substeps

        if len(self.return_grid) > 0:
            for idx, i in enumerate(self.return_grid):
                self.sim.compute_grid_mass(f, i, backward_grad=args[idx])

        n_bodies = self.sim.n_bodies
        start = -n_bodies
        dist_grads = particle_grad[:, start:]
        self.sim.get_dists(f, dist_grads)

        particle_grad = particle_grad[..., :start]
        particle_grad = particle_grad.reshape(-1, self.dim * 2)
        tool_grad = tool_grad.reshape(n_bodies, 7)

        x = particle_grad.detach().cpu().numpy()
        c = tool_grad.detach().cpu().numpy()
        state = self.sim.states[f]
        state.x_grad.cuda_add(x[:, :3])
        state.v_grad.cuda_add(x[:, 3:])
        state.body_pos_grad.cuda_add(c[:, :3])
        state.body_rot_grad.cuda_add(c[:, 3:])

    def get_diff_forward(self):
        class forward(Function):
            @staticmethod
            def forward(ctx, s, pos, rot, velocity, *past_obs):

                save_tensors = [torch.tensor([s]), *[torch.zeros_like(i) for i in past_obs]]
                ctx.save_for_backward(*save_tensors)

                ctx.velocity = velocity

                f = s * self.substeps
                if velocity is not None:
                    v = self.sim.get_v(f, 'numpy')
                    assert v.shape == velocity.shape, f"{v.shape} {velocity.shape}"
                    
                    v += velocity.detach().cpu().numpy()
                    self.sim.states[f].v.upload(v)
                    # change the velocity of the state v..

                for i in range(self.substeps):
                    self.sim.set_pose(self.sim.states[f+i+1], pos[i], rot[i], self.sim.stream0)
                    self.sim.substep(f+i, clear_grad=True)
                self.sim.sync()
                return self.get_obs(s+1, pos.device)

            @staticmethod
            def backward(ctx, *obs_grad):
                zero_grads = ctx.saved_tensors
                s = zero_grads[0].item()


                self.set_obs_grad(s+1, *obs_grad) # add the gradient back into the tensors ...
                f = s * self.substeps

                pos = []
                rot = []
                for i in range(f + self.substeps-1, f-1, -1):
                    self.sim.substep_grad(i)
                    state = self.sim.states[i+1]
                    pos.append(state.body_pos_grad.download(n=self.sim.n_bodies, stream=self.sim.stream1))
                    rot.append(state.body_rot_grad.download(n=self.sim.n_bodies, stream=self.sim.stream1))

                self.sim.sync()

                pos_grad = torch.tensor(np.array(pos[::-1]), device=self.device)
                rot_grad = torch.tensor(np.array(rot[::-1]), device=self.device)

                out = (None, pos_grad, rot_grad)
                velocity_grad = None
                if ctx.velocity is not None:
                    velocity_grad = self.sim.states[f].v_grad.download(n=self.sim.n_particles, device='cuda:0')
                out = out + (velocity_grad,)

                return out + zero_grads[1:]

        self._forward_func = forward
        self.diff_forward = self._forward_func.apply
