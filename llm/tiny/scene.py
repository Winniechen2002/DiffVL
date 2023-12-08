import numpy as np
import torch
from .mydsl import dsl, none
from .softbody import soft_body_type, SoftBody
from .tool import tool_type, Tool
from ..pl.types import DataType, List
from envs import WorldState, MultiToolEnv

myint = dsl.int
mystr = dsl.str


class Scene:
    def __init__(self, env: MultiToolEnv, collect_obs=True):
        self.env = env
        if collect_obs:
            self.initial_state = self.env.initial_state
            self.obs = [self.env.get_obs()]
        else:
            self.obs = []

        self.cur_step = 0

    

    def check_soft_bodies(self, cen, rad):
        self.initial_state.check_soft_bodies(cen, rad)
        self.env.set_state(self.initial_state)
        self.initial_state = self.env.initial_state

    # def erase_soft_bodies(self, dir, pos):
    #     self.initial_state.erase_soft_bodies(dir, pos)
    #     self.env.set_state(self.initial_state)
    #     self.initial_state = self.env.initial_state
        

    @dsl.as_attr
    def set_timestep(self, cur_step: myint) -> none:
        self.cur_step = cur_step

    def get_obs(self):
        if self.cur_step >= len(self.obs):
            import logging
            logging.warning("get_obs: cur_step < len(self.obs)")
            return self.obs[-1]
        return self.obs[self.cur_step] 

    def start(self):
        self.tables = []
        self.set_timestep(0)


    def collect(self, idx=None):
        obs = self.env.get_obs()
        if idx is None or idx >= len(self.obs):
            if idx: 
                assert idx == len(self.obs)
            self.obs.append(obs)
        else:
            self.obs[idx] = obs

    @dsl.as_attr
    def get_object_list(self) -> List(soft_body_type):
        # raise NotImplementedError 
        return [SoftBody.new(self, i) for i in np.unique(self.initial_state.ids).astype(np.int32)]

    @dsl.as_attr
    def obj(self, i: myint) -> soft_body_type:
        if i >= 0: 
            assert (self.initial_state.ids == i).any()
            return SoftBody.new(self, i)
        else:
            return SoftBody(
                self, torch.arange(
                    len(self.initial_state.ids), device='cuda:0')
            )

    @dsl.as_attr
    def tool(self) -> tool_type:
        timesteps = list(np.arange(len(self.obs)))
        return Tool(self)


    @dsl.as_attr
    def T(self) -> myint:
        return len(self.obs)



scene_type = dsl.build_data_type(Scene)