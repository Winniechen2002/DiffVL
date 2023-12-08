import torch
import copy
import tqdm
import numpy as np
from tools.utils import tonumpy, totensor, logger
from .libtool import MotionPlanner, MultiToolEnv, Scene, SoftBody
from ..tiny.libpcd import relative_pose



# https://github.com/hzaskywalker/Concept/blob/d08eece19522320f3fdb2d443789412047989fb3/solver/cpdeform/sampler.py#L20
class CPDeform(MotionPlanner):
    def __init__(self, env: MultiToolEnv, *args, minimal_gap=1./64, scale=(0., 0., 0., 1.,0., 0.)) -> None:
        super().__init__(env, *args, minimal_gap=1./64)
        assert len(args) == 2
        self.scale = totensor(scale, device='cuda:0')

    @torch.no_grad()
    def execute(self, env, scene, action, images=None):
        assert env.get_cur_steps() == 0
        if action is None:
            return
        #qpos = path[-1]
        tool = env.tool_cur

        #for i in action['path'][1:]:
        tool.set_state(action['qpos'], 0)
        pos, rot = tool.forward_kinematics(totensor(action['qpos'], device=self.env.device))
        dists = self.compute_sdf(pos, rot, 0)
        if images is not None:
            for i in range(10):
                # repeat for 10 times ..
                images.append(env.render('rgb_array'))
        return 

        
    def find_solution(self, scene):
        obj, target = self.args[:2]
        obs, qpos = self.sample_pose(scene, obj, target)

        from llm.envs.end_effector import Gripper
        if self.env.tool_cur.get_action_shape()[-1] == 7:
            out = {'qpos': qpos}
        else:
            out = Gripper.forward_kinematics(self.env.tool_cur, qpos) # this also works for Fingers ..
            from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles

            pos = out[0]
            rot = matrix_to_euler_angles(quaternion_to_matrix(out[1]), 'XYZ')
            out = {'qpos': torch.cat([pos, rot], dim=-1)}

        return out

    def sample_pose(self, scene: Scene, obj: SoftBody, target: torch.Tensor, n=100, **kwargs):
        scene.cur_step = len(scene.obs)
        obs = scene.get_obs() # use the last observation, same in Motion planner ..
        pcd = obs['pos']
        pcd = pcd[obj.indices] # cur pcd ..

        score = totensor(match(pcd, target), device='cuda:0')
        # transport_gradient = calculate_transport_gradient(pcd, target)

        center = (pcd.max(0).values + pcd.min(0).values) / 2
        half_size = (pcd.max(0).values - pcd.min(0).values) / 2


        xx = []
        for i in self.scale[3:]:
            if i > 0.:
                xx.append([-np.pi/2, 0., np.pi/2])
            else:
                xx.append([0.])
        from itertools import product
        xx = product(*xx)
        xx = totensor(list(xx), device='cuda:0')

        samples = (torch.rand(n, 6, device=self.scale.device) * 2 - 1) * self.scale[None, :]
        samples[:, :3] = center + samples[:, :3] * half_size
        samples[:, 3:] = samples[:, 3:]
        # samples[:, 3:] = xx[torch.randint(0, len(xx), size=(len(samples),))]

        weight = - np.inf
        best_qpos = None

        for obs, qpos, dists in tqdm.tqdm(self.grasp(scene, obj, None, samples=samples), total=len(samples)):
            dists = dists[obj.indices]

            topk_val = - ((-dists).topk(int(len(dists) * 0.05), dim=0).values[-1:, :])
            mask = (dists <= topk_val).float()

            if False:
                # in some cases, we hope the direction of the manipulator aligns with the transport gradient.
                tmp_dirs = np.repeat(finger_dir[None, ...], transport_gradient.shape[0], axis=0)
                dir_sim = np.abs(np.sum(tmp_dirs * transport_gradient, axis=1)) # np.dot()
                dir_sims = np.c_[dir_sim, dir_sim]
            else:
                dir_sims = 1
            mani2particle_weights = (mask * score[:, None] * dir_sims).mean()
            if mani2particle_weights > weight:
                weight = mani2particle_weights
                best_qpos = qpos
        
        save_pcd(draw_potentials_on_pts(pcd, score), 'tmp.pcd')
        self.set_qpos(best_qpos)
        logger.savefig('found.png', self.env.render('rgb_array', index=-1))
        return obs, best_qpos


        
if __name__ == '__main__':
    # test shape match
    import os
    from llm import LLMPATH
    from llm.genetic.libskill import load_shape
    path = os.path.join(LLMPATH, 'genetic', 'prototype')
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        os.system("python3 ../build_prototype.py")

    pcd1 = load_shape(os.path.join(path, 'cube.pcd'))
    pcd2 = load_shape(os.path.join(path, 'triangle.pcd'))

    color = np.zeros((2048*2, 3))
    color[2048:, 0] = 1.
    color[:2048, 1] = 1.
    save_pcd(torch.concat((pcd1, pcd2)), 'tmp.pcd', color)

    cpdeform = CPDeform(None)

    matched = cpdeform.match_shape(pcd1, pcd2)
    save_pcd(torch.concat((pcd1, matched)), 'tmp2.pcd', color)