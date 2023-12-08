# a differentiable version of the shape program
import torch
import numpy as np
from llm.pl import Executor
from .logprob import LogProb

def __Position__Eq(self, other):
    return  LogProb(0.0025 - torch.linalg.norm(other.pos - self.pos)**2)

delta = -0.00

def __Position__left(self, other):
    return LogProb(other.pos[0] - 0.05 - self.pos[0])

def __Position__right(self, other):
    return LogProb(self.pos[0] - 0.05 - other.pos[0])

def __Position__front(self, other):
    #return diff_greater_zero( delta - (self.pos[2] - other.pos[2]), scaling=0.025)
    raise NotImplementedError

def __Position__behind(self, other):
    return LogProb(other.pos[2] - 0.05 - self.pos[2])


def bind(executor: Executor):
    executor.register_op("__Position__Eq", __Position__Eq)
    executor.register_op("__Position__left", __Position__left)
    executor.register_op("__Position__right", __Position__right)

    executor.register_op("__Position__front", __Position__front)
    executor.register_op("__Position__behind", __Position__behind)