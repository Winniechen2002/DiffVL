import torch 
from typing import TypedDict


class OBSDict(TypedDict):
    pos: torch.Tensor  #["N", 3]
    vel: torch.Tensor  #["N", 3]
    tool: torch.Tensor #["M", 7] # rigid bodies (N, 7): 3 xyz + 4 quaternion
    dist: torch.Tensor #["N", "M"] # 
    qpos: torch.Tensor # qpos of the tool