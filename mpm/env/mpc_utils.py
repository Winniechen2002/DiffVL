from graphviz import render
import tqdm
import os
import torch
import numpy as np
from tools import Configurable, merge_inputs
from .solver_utils import np2th


class MPC(Configurable):
    def __init__(self, cfg=None, lr=0.05, horizon=20, num_iters=10, inherit=False, render_path=None):
        super().__init__()

    def reset(self, env, state=None, initial_actions=None, T=None, loss_fn=None, **kwargs):
        assert initial_actions is not None or T is not None, "you must provide solver either initial action or expected time steps.."


        self.cur_cfg = merge_inputs(self._cfg, **kwargs)
        from .diffenv import DifferentiablePhysicsEnv
        self.env: DifferentiablePhysicsEnv = env
        # actions = [(i.detach().cpu().numpy() if isinstance(i, torch.Tensor) else np.array(i)) for i in actions]
        if state is None:
            state = env.get_state()

        if initial_actions is None:
            initial_actions = np.array([env.manipulator.get_initial_action() for i in range(T)])

        self.action = torch.nn.Parameter(np2th(np.array(initial_actions)), requires_grad=True)
        self.optim = torch.optim.Adam([self.action], lr=self.cur_cfg.lr)
        self.cur_step = 0
        self.loss_fn = loss_fn if loss_fn is not None else env._loss_fn

        self.outputs = []

    def step(self, **kwargs):
        self.optim.zero_grad()
        state = self.env.get_state() # get the first steps ..

        cfg = merge_inputs(self.cur_cfg, **kwargs)

        last_idx = self.cur_step + cfg.horizon - 1
        if self.cur_step >= 1 and cfg.inherit and last_idx < len(self.action):
            with torch.no_grad():
                self.action.data[last_idx] = self.action.data[self.cur_step-1:last_idx].mean(dim=0)

        action = self.action[self.cur_step:last_idx+1]
        horizon = len(action)

        state['requires_grad'] = True
        it = tqdm.trange(cfg.num_iters)

        for iter_id in it:
            with torch.no_grad():
                self.optim.zero_grad()
                self.env.set_state(state, horizon=horizon)

            outputs = []

            def calc_loss(idx, observations):
                l = self.loss_fn(idx, **observations)
                if isinstance(l, tuple):
                    outputs.append(l[1])
                    return l[0]
                elif isinstance(l, dict):
                    loss = l.pop('loss')
                    outputs.append(l)
                    return loss
                else:
                    return l

            loss = 0
            for idx, a in enumerate(action):
                self.env.manipulator.step(a)
                observations = self.env._taichi_state()
                loss += calc_loss(idx, observations)

            #if horizon == cfg.horizon:
            # take the average.. to make it in consitent with solve_utils.
            #loss = loss/horizon * 50 

            loss.backward()
            self.optim.step()

            with torch.no_grad():
                action.data[:] = self.env.manipulator.postprocessing_actions(
                    torch.clamp(action, -1, 1))

                last_loss = loss.item()

                word = f"{iter_id}: {last_loss:.4f}"
                if len(outputs) > 0:
                    for i in outputs[-1]:
                        out = 0
                        for j in outputs:
                            if i in j:
                                out += j[i]
                        word += f', {i}: {out:.3f}'
                it.set_description(word, refresh=True)

        state['requires_grad'] = False
        self.env.set_state(state)
        self.env.step(self.action[self.cur_step].detach().cpu().numpy())

        self.cur_step += 1


    def mpc(
        self,
        env,
        *args,
        images=None,
        **kwargs
    ):
        self.reset(env, *args, **kwargs)
        render_path = merge_inputs(self._cfg, **kwargs).render_path

        if render_path is not None:
            os.makedirs(render_path, exist_ok=True)
        all_images = []

        def cv_write(name, img):
            import cv2
            cv2.imwrite(os.path.join(render_path, f'{name}.png'), img[..., :3][..., ::-1])

        for step in tqdm.trange(len(self.action)):
            assert step == self.cur_step
            self.step()

            if render_path is not None or images is not None:
                img = env.render('rgb_array')
                all_images.append(img)

                if images is not None:
                    images.append(img)
                if render_path is not None:
                    cv_write(f'{step:04d}', img)
                    reached = env.render('rgb_array', primitive=0)
                    cv_write('reached', reached)

        if render_path is not None:
            from ..utils import animate
            animate(all_images, os.path.join(render_path, 'final.mp4'), _return=False)

            img = env.render('rgb_array', primitive=0)
            cv_write('reached', img)

        return env.get_state()