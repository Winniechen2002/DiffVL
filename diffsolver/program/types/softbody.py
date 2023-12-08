import torch
from typing import List, Tuple
import numpy as np
from tools.utils import totensor


class SoftBody:
    # soft body is
    # def __init__(self, all_pcd: torch.Tensor, indices: torch.Tensor):
    #     # from .trajectory import Trajectory
    #     # self.scene: Trajectory = scene
    #     # self.trajs = self.scene.obs
    #     self.all_pcd = all_pcd
    #     indices = indices.to(all_pcd.device)
    #     self._pcd = all_pcd[indices]
    #     self.indices = indices

    def __init__(self, pcd: torch.Tensor, indices: torch.Tensor) -> None:
        self._pcd = pcd
        self.indices = indices
        assert len(pcd) == len(indices)

    @classmethod
    def new(cls, pcd: torch.Tensor, indices: torch.Tensor):
        return cls(pcd[indices], indices)


    def pcd(self):
        return self._pcd

    def N(self):
        return len(self.indices)