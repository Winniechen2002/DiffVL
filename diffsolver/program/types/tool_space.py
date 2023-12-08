"""
Utility functions to handle different type of tools. 
The tool space corresponds to the type of tool. However, the same tool may have different ways of sampling.
"""
import abc
import numpy as np  
import torch
from gym.spaces import Box
from numpy.typing import NDArray
from typing import List, Optional, Tuple
from pytorch3d.transforms import quaternion_apply, matrix_to_quaternion, euler_angles_to_matrix


class ToolCheckerException(Exception):
    pass


EPS = 1e-6

class ToolSpace(abc.ABC):
    sample_space: Box

    @abc.abstractmethod
    def get_sample_space(self) -> Box:
        pass

    @abc.abstractmethod
    def get_xyz(self, qpos: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_rot(self, qpos: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_xyz_index(self) -> List[int]:
        pass

    @abc.abstractmethod
    def get_rot_index(self) -> List[int]:
        pass

    def get_gap(self, qpos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_gap_index(self) -> List[int]:
        raise NotImplementedError

    def __init__(self, tool_cfg, sample_tips) -> None:
        """
        if sample_tips is True, then we will treat the tips as the center of the tool
        """
        self.sample_space = self.get_sample_space()
        self.low_range = list(self.sample_space.low)
        self.high_range = list(self.sample_space.high)
        self.tool_cfg = tool_cfg
        self.use_tips = sample_tips


    def set_values(self, key: List[int], value: List[float|Tuple[float, float]]):

        if len(key) != len(value):
            raise ToolCheckerException(f"key and value shape is not consistent: {len(key)} vs {len(value)}")

        for k, v in zip(key, value):
            if isinstance(v, float):
                assert self.low_range[k]- EPS <= v <= self.high_range[k] + EPS, f"qpos {k} is not in range: {self.low_range[k]} <= {v} <= {self.high_range[k]}"
                self.low_range[k] = self.high_range[k] = v
            elif isinstance(v, tuple):
                self.high_range[k] = min(self.high_range[k], v[1])
                self.low_range[k] = max(self.low_range[k], v[0])
                assert self.low_range[k] - EPS <= self.high_range[k], f" {self.low_range[k]} > {self.high_range[k]}"
            else:
                raise NotImplementedError(f"type {type(v)} is not supported")
        print(self.low_range, self.high_range)
            

    def sample(self, N: int, ignore_constraint=False) -> NDArray[np.float32]:
        low = np.asarray(self.low_range if not ignore_constraint else self.sample_space.low)
        high = np.asarray(self.high_range if not ignore_constraint else self.sample_space.high)
        samples = np.random.uniform(low, high, (N, *self.sample_space.shape)).astype(np.float32)
        return self.samples2qpos(samples)

    def samples2qpos(self, sample: NDArray[np.float32]) -> NDArray[np.float32]:
        return sample

        
    def rotx(self):
        return self.get_rot_index()[0]

    def roty(self):
        return self.get_rot_index()[1]
    
    def rotz(self):
        return self.get_rot_index()[2]


    def get_state_sampler(self, start_qpos: NDArray[np.float64], goal_qpos: NDArray[np.float64]):
        raise NotImplementedError
    

class Gripper(ToolSpace):
    """
    The default gripper dof is x, y, z and rotation around (x, y, z)
    The last dof is the gripper open/close
    """
    def get_sample_space(self) -> Box:
        high = np.array([1., 0.3, 1., np.pi, np.pi, np.pi, 1])
        low = np.array([0., 0., 0., -np.pi, -np.pi, -np.pi, 0.])
        return Box(low=low, high=high, shape=(7,))

        
    def get_xyz(self, qpos: torch.Tensor):
        return qpos[:3]

    def get_rot(self, qpos: torch.Tensor):
        return qpos[3:6]

    def get_xyz_index(self) -> List[int]:
        return [0, 1, 2]

    def get_rot_index(self) -> List[int]:
        return [3, 4, 5]

    def get_gap(self, qpos: torch.Tensor):
        return qpos[6]

    def get_gap_index(self) -> List[int]:
        return [6]

    def tip2qpos(self, qpos):
        if self.use_tips:
            qpos = torch.tensor(qpos)
            half = self.tool_cfg.size[1]/2
            rot = matrix_to_quaternion(euler_angles_to_matrix(qpos[..., 3:6], 'XYZ'))
            qpos[..., :3] += quaternion_apply(rot, torch.tensor([0., half * 0.5, 0.], dtype=torch.float32))
            qpos = qpos.detach().cpu().numpy()
        return qpos

    def samples2qpos(self, sample: NDArray[np.float32]) -> NDArray[np.float32]:
        qpos = super().samples2qpos(sample)
        return self.tip2qpos(qpos)

    def get_state_sampler(self, start_qpos: NDArray[np.float32], goal_qpos: NDArray[np.float32]):
        start_rot = start_qpos[3:6]
        target_rot = goal_qpos[3:6]
        def state_sampler():
            xyz = np.random.rand(3) * np.array([1., 0.3, 1.])
            rot = np.random.rand() * (target_rot - start_rot) + start_rot
            gap = np.random.rand() * (start_qpos[-1] - goal_qpos[-1]) + goal_qpos[-1]
            return np.r_[xyz, rot, gap]
        return state_sampler

        
class DoublePushers(Gripper):

    def get_xyz(self, qpos: torch.Tensor):
        return (qpos[:3] + qpos[6:9])/2

    def get_rot(self, qpos: torch.Tensor):
        return (qpos[3:6] + qpos[9:12])/2


    def get_sample_space(self) -> Box:
        high = np.array([1.1, 0.4, 1.1, np.pi, np.pi, np.pi, 4])
        low = np.array([-0.1, -0.2, -0.1, -np.pi, -np.pi, -np.pi, 0.])
        return Box(low=low, high=high, shape=(7,))

    def samples2qpos(self, sample: NDArray[np.float32]) -> NDArray[np.float32]:
        qpos = torch.tensor(sample, dtype=torch.float32)
        qpos = self.tip2qpos(qpos)
        assert not self.use_tips

        center = qpos[..., :3]
        gap = qpos[..., 6:7] * 0.1 + 0.01 * 1.2

        _rot = qpos[..., 3:6]
        rot = matrix_to_quaternion(euler_angles_to_matrix(_rot, 'XYZ'))
        dir = quaternion_apply(rot, torch.tensor([1., 0., 0], device=qpos.device, dtype=qpos.dtype))
        verticle = quaternion_apply(rot, torch.tensor([0., 1., 0], device=qpos.device, dtype=qpos.dtype))
        center = qpos[..., :3] + verticle * 0.05 * 0.5 #TODO: remove this hack


        left_pos = center - dir * gap
        right_pos = center + dir * gap
        return torch.cat((left_pos, _rot, right_pos, _rot), dim=-1).detach().cpu().numpy()

class Pusher(ToolSpace):
    def get_sample_space(self) -> Box:
        high = np.array([1., 0.4, 1., np.pi, np.pi, np.pi])
        low = np.array([0., 0., 0., -np.pi, -np.pi, -np.pi])
        return Box(low=low, high=high, shape=(6,))

        
    def get_xyz(self, qpos: torch.Tensor):
        return qpos[:3]

    def get_rot(self, qpos: torch.Tensor):
        return qpos[3:6]

    def get_xyz_index(self) -> List[int]:
        return [0, 1, 2]

    def get_rot_index(self) -> List[int]:
        return [3, 4, 5]

    def tip2qpos(self, qpos):
        if self.use_tips:
            qpos = torch.tensor(qpos)
            half = self.tool_cfg.size[1]
            rot = matrix_to_quaternion(euler_angles_to_matrix(qpos[..., 3:6], 'XYZ'))
            qpos[..., :3] += quaternion_apply(rot, torch.tensor([0., half*2, 0.], dtype=torch.float32))
            print(qpos)
            qpos = qpos.detach().cpu().numpy()
        return qpos

    def samples2qpos(self, sample: NDArray[np.float32]) -> NDArray[np.float32]:
        qpos = super().samples2qpos(sample)
        return self.tip2qpos(qpos)


    def get_state_sampler(self, start_qpos: NDArray[np.float32], goal_qpos: NDArray[np.float32]):
        start_rot = start_qpos[3:6]
        target_rot = goal_qpos[3:6]
        def state_sampler():
            xyz = np.random.rand(3) * np.array([1., 0.3, 1.])
            rot = np.random.rand() * (target_rot - start_rot) + start_rot
            return np.r_[xyz, rot]
        return state_sampler

        
class EmptySpace(ToolSpace):
    def get_sample_space(self) -> Box:
        return Box(low=0, high=0, shape=(1,))

    def get_xyz(self, qpos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_rot(self, qpos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_xyz_index(self) -> List[int]:
        raise NotImplementedError

    def get_rot_index(self) -> List[int]:
        raise NotADirectoryError