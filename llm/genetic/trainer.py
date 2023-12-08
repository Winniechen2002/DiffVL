import torch
import os
from torch import nn
import numpy as np
import tqdm
from typing import Optional
from llm.tiny import dsl, Scene, make_executor, Operator, scene_type
from tools.utils import totensor, logger, tonumpy
# from llm.envs import MultiToolEnv, MPMSimulator
from envs import MultiToolEnv, MPMSimulator
from tools.config import Configurable, merge_inputs
# from .genome import Genome, Gene
from .init_state import str2state

make_executor() # differentiable executor
diff_executor = dsl.differentiable_executor 
bool_executor = dsl.default_executor 



class Trainer(Configurable):
    def __init__(self,
                 program: Operator,
                 env: Optional[MultiToolEnv],
                 cfg=None,

                 path=None,
                 render_iters=50,  # render every 50 steps ..
                 max_iters=300,
                 lr=0.01,
                 override_state_cache=False,

                 T = None,
                 scene=None,

                 end_stage = None,
                 n_stages=None,
                 init_actions=None,

                 max_non_decrease_iters=20,

                 env_cfg=MultiToolEnv.get_default_config(
                    sim_cfg=dict(max_steps=100, gravity=(0., -0.9, 0.), ground_friction=1., n_particles=0),
                 ),
                 use_wandb=None,

                 ) -> None:
        super().__init__()

        self.program = program
        assert self.program.tp.out == scene_type

        print('\nscene config:')
        print(cfg)
        print('\nprogram config:')

        from .render import format_code, render_prog
        #print(format_code(self.program.pretty_print()), '\n')
        #with open('test.html', 'w') as f:
        #    f.write(render_prog(self.program.pretty_print()))
        print(self.program.pretty_print(), '\n')
        #exit(0)


        self.init_state = str2state(scene, override_state_cache)

        n_particles = len(self.init_state.X) + 10
        n_particles = max(n_particles, env_cfg.sim_cfg.n_particles)
        env_cfg.defrost()
        env_cfg.sim_cfg.n_particles = n_particles

        self.env = env if env is not None else MultiToolEnv(cfg=env_cfg)

        # (T+1) * 20 + 10
        self.device = 'cuda:0'
        self.env.set_state(self.init_state)

    def wrap_action(self, action):
        #TODO: support other action scales ..
        #n = self.worm_scale.nelement()
        tool = action.reshape(self.env.action_space.shape) # * self.actor_scale
        return totensor(tool, device=self.device), None


    def get_initial_action(self, T, init_action_path=None):
        self.env.extend((T+1) * 20 + 10)

        init_actions = np.zeros((T, *self.env.action_space.shape), dtype=np.float32)
        if init_action_path is not None:
            from .utils import SOLPATH
            action_dict = torch.load(os.path.join(
                SOLPATH, init_action_path.strip(), 'best_action'))
            actions = action_dict['actions']
            l = min(len(actions), len(init_actions))
            init_actions[:l] = actions[:l]
        else:
            action_dict = {}
        action_dict['actions'] = init_actions
        return action_dict

        
    @torch.no_grad()
    def get_stage_info(self):
        # run the initial actions to determine the part of action sequence that we want to optimize ..
        self.env.set_state(self.init_state, requiers_grad=False)
        scene = Scene(self.env)
        for i in range(1000):
            scene.obs.append(scene.obs[0])

        self.stage_limit = None
        scene = self.eval(scene)
        # determine the number of stages ..
        print('total_stages', len(scene.stage_timesteps))
        self.stage_timesteps = scene.stage_timesteps
        return self.stage_timesteps[-1].end


    def write_video(self, action_dict, filename=None, ending=None, start=None, end=None, render_prog=False, wandb_name=None):
        self.env.set_state(self.init_state)
        if filename is not None:
            images = []

        if len(action_dict['actions']) > 0:
            actions = totensor(action_dict['actions'], device=self.env.device)
        else:
            actions = []

        stage_id = 0
        scene = Scene(self.env)

        def act(idx):
            nonlocal stage_id
            while stage_id < len(self.stage_timesteps) and idx >= self.stage_timesteps[stage_id].start:
                stage = self.stage_timesteps[stage_id]
                if stage.no_grad():
                    stage.handler.execute(self.env, scene, action_dict.get(stage_id+1, None),
                                          images=images if start is None and filename is not None else None)
                    scene.collect(idx) # update the observation before idx ..
                stage_id += 1

        for idx, i in enumerate(actions):
            #TODO: execute the one after the action sequence ..
            act(idx)

            if ending is not None and idx >= ending:
                break

            self.env.step(*self.wrap_action(i))
            scene.collect()

            if filename is not None:
                if start is None or (idx>=start and idx < end):
                    images.append(self.env.render('rgb_array'))

        if ending is None or len(actions) < ending:
            act(len(actions))

        if render_prog:
            out, context = self.eval(
                scene,
                return_context=True,
                stage_limit=(1, self.stage_limit[0]-1)
            )
            from .render import render_prog

            data = {}
            for k, v in context.items():
                if (isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[-1] == 3):
                    data[k] = self.env.render_state_rgb(pos=v.detach().cpu().numpy())

                from llm.tiny import SoftBody
                if isinstance(v, SoftBody):
                    pos = scene.obs[min(v.init_T, len(scene.obs)-1)]['pos'][v.indices]
                    color = scene.initial_state.color[v.indices.detach().cpu().numpy()]
                    data[k] = self.env.render_state_rgb(
                        pos=pos.detach().cpu().numpy(), color=color)

            html = render_prog(
                self.program.pretty_print(),
                images=data, path=logger.get_dir())
            logger.write_html(html, 'program.html')

        if filename is not None and len(images) > 0:
            logger.animate(images, f"{filename}.mp4", wandb_name=wandb_name)

        return scene


    @torch.no_grad()
    def prepare_stage_optimizer(self, action_dict, cfg):
        stage_timesteps = self.stage_timesteps
        # note that the stage should be between 1 and len(stage_timesteps)
        if cfg.end_stage is not None:
            assert cfg.n_stages is not None
            assert cfg.end_stage <= len(stage_timesteps)

        end_stage = cfg.end_stage if cfg.end_stage is not None else len(stage_timesteps)
        n_stages = end_stage if cfg.n_stages is None else min(cfg.n_stages, end_stage)

        for i in range(n_stages):
            start_stage = end_stage - i - 1
            if stage_timesteps[start_stage].no_grad():
                break

        if cfg.end_stage is None and start_stage != 0:
            raise Exception("End stage is not specified and we will optimize all stages .."
                            "however, there exists a non differentiable stage before ..")

        start = stage_timesteps[start_stage].start
        end = stage_timesteps[end_stage].start if end_stage < len(stage_timesteps) else len(action_dict['actions'])


        self.env.set_state(self.init_state, requiers_grad=False)
        self.stage_limit = (start_stage + 1, end_stage + 1) # end_stage must be included ..

        scene = self.write_video(action_dict, filename=None, ending=start, render_prog=True) #NOTE: this is essential to get the initial state ..
        init_state = self.env.get_state()
        assert len(scene.obs) == start + 1
        
        return {
            'scene': scene,
            'init_state': init_state,
            'start': start,
            'end': end,
        }

        
    def forward(self, actions, init_state, checkpoint=False, requires_grad=False, start=None, end=None, scene=None):
        self.env.set_state(init_state, requires_grad=requires_grad)

        checkpoint = checkpoint and requires_grad
        if checkpoint:
            raise NotImplementedError("We do not support checkpoint yet ..")

        if scene is None:
            scene = Scene(self.env)
        if start is None:
            start = 0
        if end is None:
            end = len(actions)
        for i in range(start, end):
            self.env.step(*self.wrap_action(actions[i]))
            scene.collect(i+1)

        return self.eval(scene).tables


    def eval(self, scene, return_context=False, stage_limit=None):
        #return self.program(scene)
        diff_executor.clear_trace()
        scene.start()
        scene.cur_stage = 0
        scene.stage_timesteps = []

        assert len(diff_executor.trace) == 0

        diff_executor.global_context['stage_limit'] = stage_limit or self.stage_limit
        diff_executor.global_context['env'] = self.env

        return diff_executor.eval(self.program, scene, return_context=return_context)


    def init_optim(self, actions, cfg):
        self.optim = torch.optim.Adam([actions], lr=cfg.lr)

    
    def optimize(self, actions, **kwargs):
        self.optim.zero_grad()

        output = self.forward(actions[:], requires_grad=True, **kwargs)
        loss = 0.
        for i in output:
            loss = loss  + torch.relu( 0.0001 - i['value']) # only compute the loss when constraint (value > 0) is violated ..

        loss.backward()
        self.optim.step()

        with torch.no_grad():
            actions.data[:] = torch.clamp(actions, -1, 1)

        return {
            'loss': float(loss), 
            'creteria': float(loss),
            'tables': output
        }

    def print_table_func(self, x, decimal=5):
        v = float(x['value'])
        if decimal is not None:
            v = ('{:.' + str(decimal) + 'f}').format(v)
        return v

    def main_loop(self, action_dict, **kwargs):
        """
        action_dict stores the information to reproduce the action sequence ..
        """
        import copy
        action_dict = copy.copy(action_dict)

        cfg = merge_inputs(self._cfg, **kwargs)

        path = cfg.path
        task = path.split('/')[-1]
        if cfg.end_stage is not None:
            assert cfg.n_stages is not None
            path = path + f"/{cfg.end_stage}_{cfg.n_stages}"

        logger.configure(path, format_strs=['csv', 'log', 'stdout'] + ([] if not cfg.use_wandb else ['wandb']),
                         config=self._cfg, project=task, group=f'stage_{cfg.end_stage}')


        # jump over stages ..
        forward_kwargs = self.prepare_stage_optimizer(action_dict, cfg)

        if forward_kwargs['start'] == forward_kwargs['end']:
            action_dict = self.stage_timesteps[cfg.end_stage - 1].handler.optimize(
                cfg.end_stage, action_dict, **forward_kwargs)

        else:
            actions = nn.Parameter(
                totensor(action_dict['actions'], device=self.device))

            # start ..
            print('start optimize', forward_kwargs['start'], 'to', forward_kwargs['end'])

            self.init_optim(actions, cfg)
            last_action, last_loss = best_action, best_loss = tonumpy(actions.data), np.inf

            range = tqdm.trange(cfg.max_iters)
            history = []


            for optim_iter in range:
                dsl.optim_iter = optim_iter #TODO: hack

                output = self.optimize(actions, **forward_kwargs)

                from .utils import print_table
                print(print_table(output['tables'], self.print_table_func, print_mode=True))

                info = {} if 'info' not in output else output['info']

                previous_loss = last_loss
                last_loss = output['creteria']
                if np.isnan(last_loss):
                    print("MEET NAN!!")
                    break

                if last_loss < -10000:
                    print("LARGE loss, may due to the bugs", last_loss)
                    continue

                last_action = tonumpy(actions.data)

                if last_loss <= best_loss:
                    best_loss, best_action = last_loss, last_action

                if last_loss < previous_loss:
                    non_decrease_iters = 0
                else:
                    non_decrease_iters += 1

                if 'constraint' in info and info['constraint'] > 0:
                    non_decrease_iters = 0

                history.append([last_action, last_loss])

                info['loss'] = float(output['loss'])

                info['best_creteria'] = float(best_loss)
                info['creteria'] = float(last_loss)
                info['grad0'] = float(torch.linalg.norm(actions.grad.data[forward_kwargs['start']]))
                info['gradT'] = float(torch.linalg.norm(actions.grad.data[forward_kwargs['end']-1]))

                logger.logkvs(info)
                logger.dumpkvs()

                if cfg.render_iters and optim_iter % cfg.render_iters == cfg.render_iters -1:
                    action_dict['actions'] = best_action
                    self.write_video(action_dict, f'{optim_iter}', start=forward_kwargs['start'], end=forward_kwargs['end'], wandb_name='latest')


                if non_decrease_iters > cfg.max_non_decrease_iters:
                    break


            action_dict['actions'] = best_action
            self.write_video(action_dict, 'best')
            state = self.env.get_state()
            c = state.color
            c = np.stack([c%256, c//256 % 256, c//256//256], -1).astype(np.uint8)

            logger.save_pcd(state.X[:, [2, 0, 1]], 'best.pcd', color=c/255.)
            logger.torch_save(
                dict(history=history, state=self.init_state, cfg=cfg), 'optim_state')
        logger.torch_save(action_dict, 'best_action')
        return action_dict