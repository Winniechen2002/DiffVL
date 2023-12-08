import numpy as np
import torch
from envs import MultiToolEnv
from .softbody import SoftBody
from .obs import OBSDict
from typing import List
from .tool_space import ToolSpace
from ..scenes import SceneTuple


class SceneSpec:
    def __init__(
        self, 
        env: MultiToolEnv, 
        obs: List[OBSDict],
        state_tuple: SceneTuple,
        tool_space: ToolSpace,
        sample: int = 2000,
    ):
        self.env = env
        self.obs = obs
        assert len(self.obs) > 0
        
        self.state_tuple = state_tuple
        self.object_names = state_tuple.names
        self.goal_names = state_tuple.goal_names
        self.goal_obs = state_tuple.goal
        self.tool_space = tool_space
        self.cur_step = 0
        self.sample = sample

    def set_timestep(self, cur_step):
        self.cur_step = (cur_step + len(self.obs)) % len(self.obs)

    def total_steps(self):
        return len(self.obs)

    def get_obs(self) -> OBSDict:
        return self.obs[self.cur_step] 

    def start(self):
        self.tables = []

    def down_sample(self, index: torch.Tensor):
        # We don't use it now.
        if index.shape[0] > self.sample:
            return index[torch.randperm(index.shape[0])[:self.sample]]
        return index

    def name2idx(self, name, other = False):
        if name == 'all':
            return torch.arange(len(self.obs[self.cur_step]['pos']))

        # if name.startswith('~'):
        #     return torch.arange(len(self.obs[self.cur_step]['pos']))[~self.name2idx(name[1:])]
        if other:
            ret = torch.arange(len(self.obs[self.cur_step]['pos']))
            index = ~torch.isin(ret, self.name2idx(name))
            return ret[index]
        if ',' in name:
            names = name.split(',')
            return torch.cat([self.name2idx(n.strip()) for n in names])

        return torch.tensor(self.object_names[name], dtype=torch.long)

    def name2idx_goal(self, name, other = False):
        if name == 'all':
            return torch.arange(len(self.goal_obs['pos']))

        if other:
            ret = torch.arange(len(self.goal_obs['pos']))
            index = ~torch.isin(ret, self.name2idx_goal(name))
            return ret[index]
        if ',' in name:
            names = name.split(',')
            return torch.cat([self.name2idx_goal(n.strip()) for n in names])

        return torch.tensor(self.goal_names[name], dtype=torch.long)

    def obj(self, i: str, t: int|None = None):
        assert isinstance(i, str)
        t = t or self.cur_step
        return SoftBody.new(self.obs[t]['pos'], self.name2idx(i))

    def otherobj(self, i: str, t: int|None = None):
        assert isinstance(i, str)
        t = t or self.cur_step
        return SoftBody.new(self.obs[t]['pos'], self.name2idx(i, other = True))

    def goal(self, i):
        return SoftBody.new(self.goal_obs['pos'], indices=self.name2idx_goal(i))

    def othergoal(self, i):
        return SoftBody.new(self.goal_obs['pos'], self.name2idx_goal(i, other = True))

    def tool(self):
        from .tool import Tool
        return Tool(self)

    def T(self):
        return len(self.obs)

    def get_object_list(self) -> List[SoftBody]:
        return [self.obj(str(i)) for i in range(len(self.object_names))]

    @classmethod
    def from_scene_tuple(cls, env: MultiToolEnv, scene_tuple: SceneTuple, tool_space: ToolSpace, requires_grad=False, sample: int = 2000):
        """
        set the state of the environment to the state in the scene tuple and return a SceneSpec
        """
        env.set_state(scene_tuple.state, requires_grad=requires_grad)
        return cls(env, [env.get_obs()], scene_tuple, tool_space, sample)

    def update_obs(self, obs):
        self.obs[self.cur_step] = obs

    @property
    def state(self):
        return self.state_tuple.state
