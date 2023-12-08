import torch
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix
from llm.tiny.library import relative_pose, pose2rpy, pose_dist
from llm.tiny import dsl, Array, as_tensor

a = torch.randn((100, 3), device='cuda:0')

t = axis_angle_to_matrix(torch.tensor([0.0, np.pi/4, 0.], device='cuda:0')) 
b = (t @ a.T).T

@dsl.as_func
def func(a: Array, b: Array):
    return pose2rpy(relative_pose(a, b), 1)

@dsl.as_func
def func2(a: Array, b: Array):
    return pose_dist(relative_pose(a, b), as_tensor([0., 0.707, 0.]))

#print(pose2rpy(relative_pose(a, b), 2))
print(func2.pretty_print())
print(func2(a, b))