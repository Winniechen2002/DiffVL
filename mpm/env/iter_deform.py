"""[summary]
This code is used for iterative sampling contact points and deform the object

We consider the simplest heuristic:

One manipulator to sample the location that 
"""
import time
import numpy as np
import os.path as osp 
import copy
import os
from ..manual.sample_utils import SimpleSolver
from tools import merge_inputs
from .solver_utils import Solver
from tools.config import Configurable
from diffrl.env.geom import save_overlap_pcd
from diffrl.manual.sample_utils import get_pos
from . import geom
import cv2
from diffrl.manual.sample_utils import surface_reconstruction
from diffrl.env import GoalReach 
import pickle
from CY.data_collection.utils_reshape import normalize_goal, goal2init    
from pydprint import dprint




def launch_iterDeform_process(env, stateGoal, render_path, num_stage=10, max_iter=300, h=20, phi=True):

    start_state, goal = stateGoal

    task = GoalReach.parse(
        start_state, goal, parse_prefix='task')
    
    start_state = task.reset(env)
    solver = IterDeform.parse(
        max_iter=max_iter, num_stage=num_stage, lr=0.01, render_path=render_path, sampler=dict(phi=phi), parse_prefix='solver')
    solver.solve(
        env,
        start_state,
        [env.manipulator.get_initial_action() for i in range(h)],
        env._loss_fn
    )

def main_process(p, env, idx, render_dir, **kwargs):
    

    goal = p.xyz.detach().cpu().numpy()
    goal = normalize_goal(goal)
    init, goal = goal2init(env, goal)

    start = env.empty_state(init=init)
    
    # save stateGoal data
    stateGoal = [start, goal]
    path_pkl, path_pcd = f'data/{idx}.pkl', f'data/{idx}-vis.pcd'
    if not os.path.exists(path_pkl): 
        pickle.dump(stateGoal, open(path_pkl, 'wb'))  
    if not os.path.exists(path_pcd):
        save_overlap_pcd(path_pcd, {'xyz': get_pos(start), 'color':[255,0,0]}, {'xyz': goal, 'color':[0,255,0]})
    
    # launch iterDeform
    render_path = f'{render_dir}/{idx}'
    if not os.path.exists(render_path):
        launch_iterDeform_process(env, stateGoal, render_path, **kwargs)

class IterDeform(Solver): 
    # number of iterations and horizons to verify the performance of the sampled points
    def __init__(
        self,
        cfg=None,
        num_stage=1,
        render_interval=0,
        sampler=SimpleSolver.to_build(TYPE=SimpleSolver)
    ):
        super().__init__(cfg)
        self.sampler: SimpleSolver = SimpleSolver.build(cfg=sampler)


    def solve(self, env, start_state, initial_actions, loss_fn, state=None, **kwargs):
        cfg = merge_inputs(self._cfg, **kwargs)
        render_path = cfg.render_path
        if render_path is not None:
            os.makedirs(render_path, exist_ok=True)

            import matplotlib.pyplot as plt
            os.makedirs(render_path, exist_ok=True)
            plt.imshow(env.render('rgb_array'))
            plt.savefig(osp.join(render_path, 'init.png'))
            plt.imshow(env.task.render_goal('rgb_array'))
            plt.savefig(osp.join(render_path, 'goal.png'))

            init = get_pos(env.get_state())
            goal = env.task.goal
            save_overlap_pcd(osp.join(render_path, 'overlap_init_goal.pcd'), {'xyz': init, 'color':[255,0,0]}, {'xyz': goal, 'color':[0,255,0]})

        env.set_state(start_state)
        images = []
        for stage_id in range(cfg.num_stage):
            initial_state = env.get_state()

            if render_path is not None:
                stage_render_path = osp.join(render_path, f'stage{stage_id}')
                os.makedirs(stage_render_path, exist_ok=True)
            else:
                stage_render_path = None

            print(f'Start stage {stage_id}')
            if stage_render_path is not None:
                print('writing to', stage_render_path, '...')

            def get_path(name):
                if render_path is not None:
                    return osp.join(stage_render_path, name)
                return None

            #cur = initial_state['pose']
            cur = geom.vec(get_pos(initial_state), device=env.device)
            goal = geom.vec(env.task.goal, device=env.device) # only for goal conditioned tasks
            initial_tool_state = self.sampler.sample(env, cur, goal, get_path(f'F{stage_id}.png'), render_pcd=True)

            assert 'tools' in initial_state
            initial_state['tools'] = initial_tool_state       

            kwargs['render_path'] = stage_render_path
            out = Solver.solve(self, env, initial_state, copy.deepcopy(initial_actions), loss_fn, **kwargs)
            initial_state['requires_grad'] = False

            filename = get_path('best')
            stage_images = env.execute(initial_state, out['best_action'], filename=filename)

            if filename is not None:
                from diffrl.utils import animate
                animate(stage_images, filename+'.mp4')
                images += stage_images
                animate(images, osp.join(render_path, 'final.mp4'))

            '''
            print('Stablizing..')

            start = time.time()
            from ..manual.gripper_utils import move_dir

            zero_action = np.zeros_like(initial_actions[0])
            tool_state = env.get_tool_state(0)
            for i in range(200):
                env.step(zero_action)

    
            dir = tool_state[1][:3] - tool_state[0][:3]
            dir = dir/(np.linalg.norm(dir)+1e-10) * 0.005
            for i in range(200):
                # the gripper should move away from each other
                env.step(np.r_[-dir, 0, 0, 0, dir, 0, 0, 0]) 
            print(f'Stablied. used {time.time() - start} second(s).')
            '''
            
            # draw overlap between the END of each stage and GOAL
            cur = get_pos(env.get_state())
            goal = env.task.goal
            save_overlap_pcd(get_path(f'overlap_dst_goal.pcd'), {'xyz': cur, 'color':[255,0,0]}, {'xyz': goal, 'color':[0,255,0]})




def main():
    # CUDA_VISIBLE_DEVICES=0 python3 -m diffrl.env.iter_deform sphere.pkl  --solver.render_path sphere --solver.num_stage 20 --solver.max_iter 300
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("state_goal")
    parser.add_argument("--env_type", default=None)
    parser.add_argument("--h", default=20, type=int)
    args, _ = parser.parse_known_args()

    from .env_hubs import comp_env
    from . import GoalReach
    assert args.env_type is None
    import pickle
    with open(args.state_goal, 'rb') as f:
        start_state, goal = pickle.load(f)

    env = comp_env()
    task = GoalReach.parse(
        start_state, goal, parse_prefix='task', parser=parser)
    
    start_state = task.reset(env)
    solver = IterDeform.parse(
        max_iter=100, num_stage=2, lr=0.01, parser=parser, parse_prefix='solver')
    solver.solve(
        env,
        start_state,
        [env.manipulator.get_initial_action() for i in range(args.h)],
        env._loss_fn
    )


if __name__ == '__main__':
    main()
