import os
import torch
from dataclasses import dataclass
from envs.world_state import WorldState
from tools.config import CN
from ...paths import get_path
from ...utils import MultiToolEnv
from ...config import SceneConfig
from .utils import dict2CN, CN2dict, list2tuple
import numpy as np
from typing import Mapping, Any, Optional, Sequence

from numpy.typing import NDArray

@dataclass
class ObjectConfig:
    yield_stress: Optional[float] = None


class SceneTuple:

    def __init__(
        self,
        state: WorldState,
        names: Mapping[str, NDArray[np.intp]],
        goal: Any,
        config: SceneConfig,
        goal_names: Any,
    ) -> None:
        self.goal = goal
        self.state =  self.load_tool_cfg(state, config.Tool)
        self.config = config

        if len(config.rename) > 0:
            outs = {}
            for k, v in names.items():
                outs[config.rename.get(k, k)] = v
            names = outs

        self.names = names
        self.goal_names = goal_names or names

        # load object cfg
        #assert isinstance(config.Objects, DictConfig)
        for k, v in config.Objects.items():
            cfg = ObjectConfig(**v)
            print(cfg)
            if cfg.yield_stress is not None:
                assert self.state.E_nu_yield is not None
                k = str(k)
                self.state.E_nu_yield[self.names[k], 2] = cfg.yield_stress


    def load_tool_cfg(self, state: WorldState, tool_kwargs: Any):
        if 'tool_name' not in tool_kwargs:
            tool_kwargs['tool_name'] = state.tool_name
        tool_kwargs = list2tuple(tool_kwargs) # some hacks
        if 'qpos' in tool_kwargs:
            tool_kwargs['qpos'] = list(tool_kwargs['qpos']) # qpos has to be a list
        return state.switch_tools(**tool_kwargs)

    def get_goal_image(self) -> Optional[NDArray]:
        return None
    
from .utils import parse_dict
def load_goal(env: MultiToolEnv, goal: str):
    assert env is not None
    if goal.endswith('.yaml'):
        goal_path = os.path.join(get_path('INIT_CFG_DIR'), goal)
        kwargs = CN(parse_dict(CN._load_cfg_from_file(open(goal_path, 'r'))))
        goal_state = WorldState.sample_shapes(**kwargs['Shape'])
        env.set_state(goal_state, requires_grad=False)
        goal_obs = env.get_obs()
        goal_image = env.render('rgb_array')
        goal_name = None
    else:
        from .visiontask.taskseq import TaskSeq
        task_str = goal.split('.')[0].split('_')
        a, b = task_str[:2]
        b = int(b)
        task = TaskSeq(int(a))
        goal_state, goal_name = task.fetch_stage(b)
        env.set_state(goal_state, requires_grad=False)
        goal_obs = env.get_obs()
        goal_image = env.render('rgb_array')
    return goal_obs, goal_image, goal_name
    
class YamlStateTuple(SceneTuple):
    def __init__(self, env: MultiToolEnv|None, config: SceneConfig) -> None:

        cfg_path = os.path.join(get_path('INIT_CFG_DIR'), config.path)
        kwargs = CN(parse_dict(CN._load_cfg_from_file(open(cfg_path, 'r'))))
        kwargs.merge_from_other_cfg(dict2CN(config.Shape))

        state = WorldState.sample_shapes(**kwargs['Shape'])
        names = {str(int(k)): np.where(state.ids==k)[0] for k in np.unique(state.ids)}
        if len(config.Tool) == 0:
            config.Tool = CN2dict(kwargs['Tool'])


        goal_image, goal_obs, goal_name = None, None, None
        if config.goal != 'none':
            assert env is not None
            goal_obs, goal_image, goal_name = load_goal(env, config.goal)
            
        self.goal_image = goal_image
        super().__init__(state, names, goal_obs, config, goal_name)

    def get_goal_image(self) -> NDArray | None:
        return self.goal_image


class TaskSeqTuple(SceneTuple):
    def __init__(self, env: MultiToolEnv, config: SceneConfig) -> None:
        from .visiontask.taskseq import TaskSeq
        name = config.path

        task_str = name.split('.')[0].split('_')

        a, b = task_str[:2]
        goal_step = int(task_str[2]) if len(task_str) > 2 else 1
        b = int(b)
        task = TaskSeq(a)
        state, names = task.fetch_stage(b)

        if config.goal != 'none':
            goal_obs, self.goal_image, goal_name = load_goal(env, config.goal)
        else:
            goal_state, goal_name = task.fetch_stage(b+goal_step)
            env.set_state(goal_state, requires_grad=False)
            goal_obs = env.get_obs()
            self.goal_image = env.render('rgb_array')
        super().__init__(state, names, goal_obs, config, goal_names=goal_name)
    
    def get_goal_image(self) -> Optional[NDArray]:
        return self.goal_image

        
class TrajLoader(SceneTuple):
    def __init__(self, state: WorldState, names: Mapping[str, NDArray[np.intp]], goal: Any, config: SceneConfig, goal_name: Any) -> None:
        super().__init__(state, names, goal, config, goal_name)

    @classmethod
    def load_from_trajs(cls, env: MultiToolEnv, config: SceneConfig):
        #TODO: organize the data structure to store data

        trajs = torch.load(config.path)
        states: Sequence[WorldState] = trajs['states']
        names: Mapping[str, NDArray[np.intp]] = trajs['names']

        init_state = states[config.state_id]
        init_state.V[:] = 0
        # init_state.F[:] = np.eye(3)
        # init_state.C[:] = 0

        goal_image, goal_obs, goal_name = None, None, None
        if config.goal != 'none':
            goal_obs, goal_image, goal_name = load_goal(env, config.goal)
        
        # if len(config.Tool) == 0:
        config.Tool = {'tool_name': init_state.tool_name, 'qpos': init_state.qpos.tolist()}
        out = cls(init_state, names, goal_obs, config, goal_name)

        setattr(out, 'goal_image', goal_image)
        return out

    def get_goal_image(self) -> Optional[NDArray]:
        return getattr(self, 'goal_image', None)
        


def load_scene_with_envs(env: Optional[MultiToolEnv], config: SceneConfig) -> SceneTuple:
    name = config.path
    if name.endswith('.yml') or name.endswith('.yaml'):
        return YamlStateTuple(env, config)

    assert env is not None
    if name.endswith('.pt'):
        return TrajLoader.load_from_trajs(env, config)
    else:
        return TaskSeqTuple(env, config)