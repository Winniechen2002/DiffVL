# def solve_grasp(env, tool_cls, start_qpos, obs, indices, checker, largest_gap=False, reset):
# grasp part/objects/points
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from ..pl import *
from envs import MultiToolEnv
from envs.end_effector import Gripper
from llm.tiny import Scene, SoftBody
from tools.utils import totensor, tonumpy, animate, logger
from tools.utils.rrt import RRTConnectPlanner


class MotionPlanner:
    def __init__(self, env: MultiToolEnv, *args, minimal_gap=1./64) -> None:
        self.env = env
        self.args = args
        self.minimal_gap = minimal_gap

    @torch.no_grad()
    def execute(self, env, scene, action, images=None):
        assert env.get_cur_steps() == 0
        if action is None:
            return
        #qpos = path[-1]
        tool = env.tool_cur

        for i in action['path'][1:]:
            tool.set_state(i, 0)
            pos, rot = tool.forward_kinematics(totensor(i, device=self.env.device))
            dists = self.compute_sdf(pos, rot, 0)
            if images is not None:
                images.append(env.render('rgb_array'))
        return 

    def find_solution(self, scene):
        obj, dir = self.args
        dir = torch.stack(dir)
        for obs, qpos, dists in self.grasp(scene, obj, dir):
            out = {'qpos': qpos}
            out['path'] = self.mp(obs, qpos, self.minimal_gap)
        return out

    @torch.no_grad()
    def optimize(self, stage_id, action_dict, **kwargs):
        self.env.set_state(kwargs['init_state'], requires_grad=False)
        scene = kwargs['scene']
        if scene is None:
            scene = Scene(self.env)
        img = self.env.render('rgb_array')
        logger.savefig('init.png', im=img)

        self.preprocess_state()

        action_dict = copy.copy(action_dict)
        action_dict[stage_id] = self.find_solution(scene)

        self.postprocess_state()
        return action_dict

    def preprocess_state(self):
        self.tmp_state_f = len(self.env.simulator.states) - 1 
        self.tmp_state = self.env.simulator.states[-1]
        self.tmp_state_backup = self.tmp_state.get_state()

    def postprocess_state(self):
        self.tmp_state.set_state(self.tmp_state_backup)

    def compute_sdf(self, pos, rot, f = None):
        #state = self.tmp_state if state is None else state
        f = self.tmp_state_f if f is None else f
        state = self.env.simulator.states[f]
        state.body_pos.upload(tonumpy(pos))
        state.body_rot.upload(tonumpy(rot))
        return self.env.simulator.get_dists(f, grad=None, device=pos.device)

    @torch.no_grad()
    def grasp(self, scene: Scene, obj: SoftBody, dir: torch.Tensor, minimal_gap=1./64, samples=None):
        self.env.simulator.set_state(-1, self.env.simulator.get_state(0))

        tool = self.env.tool_cur

        scene.cur_step = len(scene.obs)
        obs = scene.get_obs() # use the last stage ..


        pcd = obs['pos']

        center = pcd[obj.indices].mean(axis=0)
        if samples is None:
            dir = dir * np.pi / 2
            samples = [torch.cat((center, dir))]

        for state in samples:
            center, dir = state[:3], state[3:]

            for gap in range(20):
                gap = gap / 10.

                qpos = torch.zeros(7, device=center.device, dtype=center.dtype)
                #qpos = torch.zeros((center, rot, gap), device=center.device)
                qpos[:3] = center
                qpos[3:6] = dir
                qpos[6] = gap

                dists = self.set_qpos(qpos)
                if dists.min() > minimal_gap:
                    break
            yield obs, qpos, dists

    def set_qpos(self, qpos):
        tool = self.env.tool_cur
        qpos = totensor(qpos, device=self.env.device)
        pos, rot = Gripper.forward_kinematics(tool, qpos) # this also works for Fingers ..
        dists = self.compute_sdf(pos, rot)
        return dists


    def mp(self, obs, target_qpos, minimal_gap, expand_dis=0.02, step_size=0.01, max_iter=20000, use_lvc=True):
        tool = self.env.tool_cur

        def render(*args, **kwargs):
            return self.env.render(*args, **kwargs, index=self.tmp_state_f)

        dists = self.set_qpos(obs['qpos'])
        assert dists.min() > minimal_gap
        img = render('rgb_array')

        self.set_qpos(target_qpos)
        img2 = render('rgb_array')
        plt.imshow(np.concatenate([img, img2], axis=1))
        logger.savefig('xx.png')

        start_pose = tonumpy(obs['qpos'])
        target_pose = tonumpy(target_qpos)

        start_rot = start_pose[3:6]
        target_rot = target_pose[3:6]

        def sample_state():
            assert isinstance(tool, Gripper)
            xyz = np.random.rand(3) * np.array([1., 0.4, 1.]) + start_pose.clip(-np.inf, 0.)[:3]
            rot = np.random.rand() * (target_rot - start_rot) + start_rot
            gap = np.random.rand() * (start_pose[-1] - target_pose[-1]) + target_pose[-1]
            return np.r_[xyz, rot, gap]

        def collision_checker(qpos):
            dists = self.set_qpos(qpos)
            return dists.min() < minimal_gap
        planner = RRTConnectPlanner(sample_state, collision_checker, expand_dis, step_size,
                                    max_iter, 600 if use_lvc else 0)

        print(start_pose, target_pose)

        assert not collision_checker(start_pose)
        assert not collision_checker(target_pose)

        path = planner(start_pose, target_pose, info=True)

        
        def render_videos(path):
            images = []
            for i in path:
                self.set_qpos(i)
                images.append(render('rgb_array'))
            return images

        # interpolate a new path
        new_paths = []
        for idx in range(len(path)-1):
            a = path[idx]
            b = path[idx + 1]

            length = np.linalg.norm(b - a)
            j = 0
            while j < length:
                new_paths.append(a + (b - a) * j / length)
                j += 0.05

        new_paths.append(path[-1])

        logger.animate(render_videos(new_paths), 'planned.mp4')
        return new_paths