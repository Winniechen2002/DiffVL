import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
from tools import Configurable, merge_inputs
from .env_module import EnvModule

ENV = None
FUNCS = {}

DEFAULT_DTYPE = torch.float32


def set_default_tensor_type(dtypecls):
    global DEFAULT_DTYPE
    th.set_default_tensor_type(dtypecls)
    if dtypecls is th.DoubleTensor:
        DEFAULT_DTYPE = torch.float64
    else:
        DEFAULT_DTYPE = torch.float32

def np2th(nparr, device='cuda:0'):
    dtype = DEFAULT_DTYPE if nparr.dtype in (np.float64, np.float32) else None
    return th.from_numpy(nparr).to(device=device, dtype=dtype)


def np2th_cpu(nparr):
    dtype = DEFAULT_DTYPE if nparr.dtype in (np.float64, np.float32) else None
    return th.from_numpy(nparr).to(dtype=dtype)



class Solver(Configurable, EnvModule):
    def __init__(self,
                 cfg=None,
                 lr=0.01,
                 max_iter=200,
                 verbose=True,
                 early_stop=None,
                 compute_loss_in_end=False,

                 optim_type='Adam',
                 render_interval=50,
                 render_path=None,
                 save_state=1,
                 ):

        Configurable.__init__(self)


    def save_results(self, env, optim_state, action, name, cfg):
        from .diffenv import DifferentiablePhysicsEnv
        env: DifferentiablePhysicsEnv

        os.makedirs(cfg.render_path, exist_ok=True)
        video_path = os.path.join(cfg.render_path, f'{name}.mp4')
        print('rendering to', video_path, '...')

        env.execute(optim_state['initial_state'], action, filename=video_path)

        if cfg.save_state:
            torch.save(optim_state, os.path.join(cfg.render_path, 'optim_state.pkl'))

        img = env.render('rgb_array', primitive=0)
        plt.imshow(img/255.)
        plt.savefig(os.path.join(cfg.render_path, 'reached.png'))

    def solve(self, env, initial_state, initial_actions, loss_fn, state=None, **kwargs):
        from .diffenv import DifferentiablePhysicsEnv
        env: DifferentiablePhysicsEnv

        cfg = merge_inputs(self._cfg, **kwargs)
        lr = cfg.lr
        optim_type = cfg.optim_type

        if state is None:
            assert initial_actions is not None
            iter_id = 0
            optim_buffer = []
            action = torch.nn.Parameter(np2th(np.array(initial_actions)), requires_grad=True)
            if optim_type == 'Adam':
                optim = torch.optim.Adam([action], lr=lr)
            elif optim_type == 'SGD':
                optim = torch.optim.SGD([action], lr=lr)
            elif optim_type == 'RMSprop':
                optim = torch.optim.RMSprop([action], lr=lr)
            elif optim_type == 'LBFGS':
                optim = torch.optim.LBFGS([action], lr=lr)
            else:
                raise NotImplementedError
            initial_state['requires_grad'] = True
            # scheduler = scheduler if scheduler is None else scheduler(optim)
            last_action, last_loss = best_action, best_loss = initial_actions, np.inf
        else:
            iter_id = state['iter_id']
            optim_buffer = state['optim_buffer']
            action = state['action']
            optim = state['optim']
            initial_state = state['initial_state']
            # scheduler = state['scheduler']
            best_action, best_loss = state['best_action'], state['best_loss']
            last_action, last_loss = state['last_action'], state['last_loss']

        ran = tqdm.trange if cfg.verbose else range
        it = ran(iter_id, iter_id+cfg.max_iter)


        non_decrease_iters = 0
        def get_state():
            return {
                'best_loss': best_loss,
                'best_action': best_action,
                'last_loss': last_loss,
                'last_action': last_action,
                'iter_id': iter_id,
                'optim_buffer': optim_buffer,
                'action': action,
                'optim': optim,
                'initial_state': initial_state,
            }


        for iter_id in it:
            optim.zero_grad()

            loss = 0
            outputs = []
            observations = env.set_state(initial_state, horizon=len(action))

            if cfg.compute_loss_in_end:
                obs_array = []

            def calc_loss(idx, observations):
                l = loss_fn(idx, **observations)
                if isinstance(l, tuple):
                    outputs.append(l[1])
                    return l[0]
                elif isinstance(l, dict):
                    loss = l.pop('loss')
                    outputs.append(l)
                    return loss
                else:
                    return l

            for idx, i in enumerate(action):
                #observations = func.forward(idx, i, *observations)
                #env.step(i) # env step ..
                env.manipulator.step(i)
                observations = env._get_obs()

                if not cfg.compute_loss_in_end:
                    loss += calc_loss(idx, observations)
                else:
                    obs_array.append(observations)

            if cfg.compute_loss_in_end:
                for idx, observations in enumerate(obs_array):
                    loss += calc_loss(idx, observations)


            loss.backward()
            optim.step()

            #if action.grad is not None:
            #    g = action.grad.detach().cpu().numpy()
            #    if np.isnan(g).any():
            #        from IPython import embed; embed()

            with torch.no_grad():
                action.data[:] = env.manipulator.postprocessing_actions(torch.clamp(action, -1, 1))

                last_loss = loss.item()
                if np.isnan(last_loss):
                    print("MEET NAN!!")
                    break
                if last_loss < -10000:
                    print("LARGE loss, may due to the bugs", loss)
                    continue
                last_action = action.data.detach().cpu().numpy()
                if last_loss < best_loss:
                    best_loss = last_loss
                    best_action = last_action
                    non_decrease_iters = 0
                else:
                    non_decrease_iters += 1

            optim_buffer.append({'action':last_action, 'loss':last_loss})
            if cfg.verbose:
                word = f"{iter_id}: {last_loss:.4f}  {best_loss:.3f}"
                if len(outputs) > 0:
                    for i in outputs[-1]:
                        out = 0
                        for j in outputs:
                            if i in j:
                                out += j[i]
                        word += f', {i}: {out:.3f}'
                it.set_description(word, refresh=True)

            if cfg.render_path is not None and cfg.render_interval != 0 and iter_id % cfg.render_interval == 0:
                self.save_results(env, get_state(), last_action, str(iter_id), cfg)

            if cfg.early_stop is not None and cfg.early_stop and non_decrease_iters >= cfg.early_stop:
                break

        final_state = get_state()
        if cfg.render_path is not None:
            self.save_results(env, final_state, last_action, 'best', cfg)

        return final_state