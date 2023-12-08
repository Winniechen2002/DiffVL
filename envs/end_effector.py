import torch
import numpy as np
from mpm.simulator import MPMSimulator
from tools import Configurable, as_builder, merge_inputs
from mpm.utils import rigid_body_motion
from pytorch3d.transforms import quaternion_apply


class Tool(Configurable): 

    def __init__(
        self, simulator,
        cfg=None,
        friction=20.,
        K=0.0, # stiffness
        color=(0.3, 0.3, 0.3),
        softness=666.,

        lv=(0.01, 0.01, 0.01), # linear velocity
        av=(0.01, 0.01, 0.01), # angular velocity
    ):
        super().__init__()
        # mu is the friction
        self.simulator: MPMSimulator = simulator
        self.name: str = None
        self.substeps = self.simulator.substeps
        self.init()

    def init(self):
        self.tool_lists = {k: [] for k in
                           ['types', 'softness', 'mu', 'K', 'args', 'pos', 'rot', 'action_scales']}

    @classmethod
    def get_qshape(cls): # state shape
        return (cls.n_bodies, 7)

    @classmethod
    def get_action_shape(cls):
        return (cls.n_bodies * 6,)

    def check_action(self, action):
        assert action.max() < 1.1 and action.min() > -1.1
        assert action.shape == self.get_action_shape(), "{} != {}".format(action.shape, self.get_action_shape())

    def add_tool(self, type, args, pos=None, rot=None, **kwargs):
        # print(list(self.tool_lists.keys()))
        def append(key, val):
            self.tool_lists[key].append(val)

        if isinstance(type, str):
            type = {
                'Box': 0, # (x, y, z)
                'Capsule': 1, # (r, h)
            }[type]

        params = np.zeros(4)
        for idx, i in enumerate(args):
            params[idx] = i

        cfg = merge_inputs(self._cfg, **kwargs)
        append('types', type)
        # print(cfg.softness)
        append('softness', cfg.softness)
        append('mu', cfg.friction)
        append('K', cfg.K)
        append('args', params)
        append('pos', (0., 0., 0.) if pos is None else pos)
        append('rot', (1., 0., 0., 0.) if rot is None else rot)
        append('action_scales', cfg.lv + cfg.av)

    def reset(self, cfg=None):
        if cfg is not None:
            self._cfg.merge_from_other_cfg(cfg)
            self.init()

        #self.simulator.n_bodies = len(self.tool_lists['types'])
        #print("tool_lists:",self.tool_lists)
        self.simulator.n_bodies = self.n_bodies
        assert self.simulator.n_bodies == self.n_bodies
        assert self.simulator.n_bodies <= self.simulator.MAX_N_BODIES, f"{self.simulator.n_bodies}, {self.simulator.MAX_N_BODIES}"
        self.simulator.init_bodies(**self.tool_lists)

    def set_qpos(self, cur, val):
        """
        for i in range(cur+1, cur+self.substeps+1):
            if i > len(self.qpos_list):
                self.qpos_list.append(val[i-cur-1])
            else:
        self.qpos_list[cur+1:cur+self.substeps+1] = val
        """
        if cur + 1 == len(self.qpos_list):
            # for backward..
            self.qpos_list = torch.concat((self.qpos_list, val))
        else:
            assert cur == 0
            self.qpos_list = torch.concat((self.qpos_list[:1], val))

    def copy_frame(self):
        self.qpos_list[0] = self.qpos_list[self.substeps]

    def qstep(self, cur, action):
        torch_scale = self.simulator.get_torch_scale(action.device)
        substeps = self.substeps

        assert cur % substeps == 0
        action = action.reshape(-1, 6).clamp(-2., 2.) * torch_scale

        posrot = self.qpos_list[cur]
        pos = posrot[..., :3]
        rot = posrot[..., 3:7]

        pos, rot = rigid_body_motion((pos, rot),
            action[None,:].expand(substeps, -1, -1) * (
                torch.arange(substeps, device=action.device)[:, None, None]+1
                )/substeps)

        self.set_qpos(cur, torch.concat((pos, rot), axis=-1))
        
    def forward_kinematics(self, qpos):
        pos = qpos[..., :3]
        rot = qpos[..., 3:7]
        return pos, rot

    def get_state(self, index=0):
        return self.qpos_list[index].detach().cpu().numpy()

    def set_state(self, state, index=0):
        assert index == 0
        from tools.utils import totensor
        self.qpos_list = totensor(state, 'cuda:0')[None,:]

    @classmethod
    def empty_state(cls):
        #return np.zeros(self.qshape)
        qshape = cls.get_qshape()
        assert len(qshape) == (2,) and qshape[-1] == 7
        p = np.zeros(qshape)
        p[:, 3] = 1
        return p

    def add_tool_by_mode(self):
        #print("cfg-------",self._cfg)
        if self._cfg.mode == 'Box':
            self.add_tool('Box', (self._cfg.size[0], self._cfg.size[1], self._cfg.size[2]))
        else:
            self.add_tool('Capsule', self._cfg.size)


class Gripper(Tool):
    n_bodies = 2
    def __init__(self, simulator, cfg=None, size=(0.02, 0.15, 0.02),
                 action_scale=(0.015, 0.015, 0.015, 0.05, 0.05, 0.05, 0.015), friction=10., mode='Box'):
        super().__init__(simulator)

    def init(self):
        super().init()
        self.add_tool_by_mode()
        self.add_tool_by_mode()
        self._torch_scale = torch.tensor(
            np.array(self._cfg.action_scale),
            device='cuda:0', dtype=torch.float32
        )


    @classmethod
    def get_qshape(cls):
        return (3+3+1,) # center pos, y rotation, gap

    def get_action_shape(self):
        return (3+3+1,)

    @classmethod
    def empty_state(cls):
        p = np.zeros(7,)
        p[:3] = np.array([0.5, 0.15, 0.5])
        return p
    
    def qstep(self, cur, action):
        self.check_action(action)

        start_q = self.qpos_list[cur]

        pos = start_q[:3]
        rot = start_q[3:6]
        gap = start_q[-1]

        substep_ratio = ((torch.arange(self.substeps, device=action.device)[:, None]+1)/self.substeps)
        pos = pos + action[None, :3] * self._torch_scale[:3] * substep_ratio
        rot = ((rot + action[None, 3:6] * self._torch_scale[3:6] * substep_ratio) + np.pi) % (2 * np.pi) - np.pi
        gap = gap + action[None, 6] * self._torch_scale[6] * substep_ratio
        gap = torch.relu(gap)  # gap must be greater than zero.
        self.set_qpos(cur, torch.cat([pos, rot, gap], -1))

    def forward_kinematics(self, qpos):
        #print("qpos",qpos.shape)
        #pos = torch.stack(pos - gap * ) 
        _rot = qpos[..., 3:6]
        gap = qpos[..., 6:7] * 0.1 + self._cfg.size[0] * 1.2

        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        assert _rot.shape[-1] == 3, _rot.shape
        rot = matrix_to_quaternion(euler_angles_to_matrix(_rot, 'XYZ'))

        dir = quaternion_apply(rot, torch.tensor([1., 0., 0], device=qpos.device, dtype=qpos.dtype))
        verticle = quaternion_apply(rot, torch.tensor([0., 1., 0], device=qpos.device, dtype=qpos.dtype))

        center = qpos[..., :3] + verticle * self._cfg.size[1] * 0.5

        left_pos = center - dir * gap
        right_pos = center + dir * gap
        pos_list = torch.stack((left_pos, right_pos), axis=-2)
        rot_list = torch.stack((rot, rot), axis=-2)
        #print("plrl",pos_list.shape,rot_list.shape)
        return pos_list, rot_list

        

class DoublePushers(Gripper):
    # the action scale is controlled by lv..
    n_bodies = 2

    def __init__(self, simulator, cfg=None, action_scale=(0.015, 0.015, 0.015, 0.05, 0.05, 0.05), size=(0.03, 0.2, 0.2), lv=(0.012, 0.012, 0.012), friction=1., softness=666., K=0.0, mode='box'):
        super().__init__(simulator)
        self.add_tool_by_mode()
        self.add_tool_by_mode()
        self._torch_scale = torch.tensor(
            np.array(self._cfg.action_scale),
            device='cuda:0', dtype=torch.float32
        )


    def init(self):
        super().init()
        self._torch_scale = torch.tensor(
            np.array(self._cfg.action_scale),
            device='cuda:0', dtype=torch.float32
        )

    @classmethod
    def get_qshape(cls): # state shape
        return (cls.n_bodies * 6, ) # ?

    @classmethod
    def get_action_shape(cls):
        return (cls.n_bodies * 6, )

    @classmethod
    def empty_state(cls):
        qshape = cls.get_qshape()
        return np.zeros(qshape)

    def qstep(self, cur, action):
        self.check_action(action)
        action = action.reshape(-1, 6)


        _start_q = self.qpos_list[cur]
        start_q = _start_q.reshape(*_start_q.shape[:-1], -1, 6)
        pos = start_q[:, :3]
        rot = start_q[:, 3:6]
        substep_ratio = ((torch.arange(self.substeps, device=action.device)[:, None, None]+1)/self.substeps)
        pos = pos[None, :] + action[None, :, :3] * self._torch_scale[:3] * substep_ratio
        rot = ((rot[None, :] + action[None, :, 3:6] * self._torch_scale[3:6] * substep_ratio) + np.pi) % (2 * np.pi) - np.pi

        out = torch.cat([pos, rot], -1)
        out = out.reshape(-1, 12)
        assert out.shape[1:] == _start_q.shape, f"{out.shape}, {_start_q.shape}"
        self.set_qpos(cur, out)

    def forward_kinematics(self, qpos):
        #print("qpos2 ",qpos.shape)
        qpos = qpos.reshape(*qpos.shape[:-1], -1, 6)
        assert qpos.shape[-2] == 2, qpos.shape
        assert qpos.shape[-1] == 6, qpos.shape

        _rot = qpos[..., 3:6]
        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        rot = matrix_to_quaternion(euler_angles_to_matrix(_rot, 'XYZ'))
        pos = qpos[..., :3].clone()
        # pos[..., 1] += self._cfg.size[1]
        return pos, rot


class Pusher(Tool):
    # the action scale is controlled by lv..
    n_bodies = 1

    def __init__(self, simulator, cfg=None, action_scale=(0.015, 0.015, 0.015, 0.05, 0.05, 0.05), size=(0.03, 0.2, 0.2), lv=(0.012, 0.012, 0.012), friction=1., softness=666., K=0.0, mode='box'):
        super().__init__(simulator)

    def init(self):
        super().init()
        self.add_tool_by_mode()
        self._torch_scale = torch.tensor(
            np.array(self._cfg.action_scale),
            device='cuda:0', dtype=torch.float32
        )

    @classmethod
    def get_qshape(cls): # state shape
        return (6, )

    @classmethod
    def get_action_shape(cls):
        return (cls.n_bodies * 6,)

    @classmethod
    def empty_state(cls):
        qshape = cls.get_qshape()
        return np.zeros(qshape)

    def qstep(self, cur, action):
        self.check_action(action)

        _start_q = self.qpos_list[cur]
        start_q = _start_q.reshape(*_start_q.shape[:-1], -1, 6)
        pos = start_q[:,:3]
        rot = start_q[:,3:6]
        substep_ratio = ((torch.arange(self.substeps, device=action.device)[:, None, None]+1)/self.substeps)
        pos = pos + action[None, :3] * self._torch_scale[:3] * substep_ratio
        rot = ((rot[:] + action[None, 3:6] * self._torch_scale[3:6] * substep_ratio) + np.pi) % (2 * np.pi) - np.pi
        assert pos.shape[1] == 1, pos.shape

        out = torch.cat([pos, rot], -1)
        out = out.reshape(-1, 6)
        assert out.shape[1:] == _start_q.shape, f"{out.shape}, {_start_q.shape}"
        self.set_qpos(cur, out)
        
    def forward_kinematics(self, qpos):
        assert qpos.shape[-1] == 6, qpos.shape
        # qpos = qpos.reshape(-1, 6)
        # assert qpos.shape[0] == 1, qpos.shape
        qpos = qpos[..., None, :]
        _rot = qpos[..., 3:6]
        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        rot = matrix_to_quaternion(euler_angles_to_matrix(_rot, 'XYZ'))
        pos = qpos[..., :3].clone()
        return pos, rot



class Knife(Tool):
    # the action scale is controlled by lv..
    n_bodies = 1

    def __init__(self, simulator, cfg=None, action_scale=(0.015, 0.015, 0.015, 0.05, 0.05, 0.05), size=(0.001, 0.1, 0.2), lv=(0.012, 0.012, 0.012), friction=0.1, softness=666., K=0.0, mode='box'):
        super().__init__(simulator)

    def init(self):
        super().init()
        self.add_tool_by_mode()
        self._torch_scale = torch.tensor(
            np.array(self._cfg.action_scale),
            device='cuda:0', dtype=torch.float32
        )

    @classmethod
    def get_qshape(cls): # state shape
        return (cls.n_bodies, 6)

    @classmethod
    def get_action_shape(cls):
        return (cls.n_bodies * 6,)

    @classmethod
    def empty_state(cls):
        qshape = cls.get_qshape()
        return np.zeros(qshape)

    def qstep(self, cur, action):
        self.check_action(action)

        start_q = self.qpos_list[cur]
        pos = start_q[:,:3]
        rot = start_q[:,3:6]
        substep_ratio = ((torch.arange(self.substeps, device=action.device)[:, None, None]+1)/self.substeps)
        pos = pos + action[None, :3] * self._torch_scale[:3] * substep_ratio
        rot = ((rot[:] + action[None, 3:6] * self._torch_scale[3:6] * substep_ratio) + np.pi) % (2 * np.pi) - np.pi
        self.set_qpos(cur, torch.cat([pos, rot], -1))
        
    def forward_kinematics(self, qpos):
        _rot = qpos[..., 3:6]
        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        rot = matrix_to_quaternion(euler_angles_to_matrix(_rot, 'XYZ'))
        pos = qpos[..., :3].clone()
        return pos, rot

class Rolling_Pin(Tool):
    # the action scale is controlled by lv..
    n_bodies = 1

    def __init__(self, simulator, cfg=None, size=(0.2, 0.1), lv=(0.012, 0.012, 0.012), friction=0.1, softness=666., K=0.0, mode='Capsule'):
        super().__init__(simulator)

    def init(self):
        super().init()
        self.add_tool_by_mode()
        self._torch_scale = torch.tensor(
            np.array(self._cfg.action_scale),
            device='cuda:0', dtype=torch.float32
        )

    @classmethod
    def get_qshape(cls): # state shape
        return (cls.n_bodies, 6)

    @classmethod
    def get_action_shape(cls):
        return (cls.n_bodies * 6,)

    @classmethod
    def empty_state(cls):
        qshape = cls.get_qshape()
        return np.zeros(qshape)

    def qstep(self, cur, action):
        self.check_action(action)

        start_q = self.qpos_list[cur]
        pos = start_q[:,:3]
        rot = start_q[:,3:6]
        substep_ratio = ((torch.arange(self.substeps, device=action.device)[:, None, None]+1)/self.substeps)
        pos = pos + action[None, :3] * self._torch_scale[:3] * substep_ratio
        rot = ((rot[:] + action[None, 3:6] * self._torch_scale[3:6] * substep_ratio) + np.pi) % (2 * np.pi) - np.pi
        self.set_qpos(cur, torch.cat([pos, rot], -1))
        
    def forward_kinematics(self, qpos):
        _rot = qpos[..., 3:6]
        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        rot = matrix_to_quaternion(euler_angles_to_matrix(_rot, 'XYZ'))
        pos = qpos[..., :3].clone()
        return pos, rot


        
FACTORY = {
    'Gripper': Gripper,
    'DoublePushers': DoublePushers,
    'Pusher': Pusher,
    'Knife': Knife,
    'Rolling_Pin': Rolling_Pin,
}
