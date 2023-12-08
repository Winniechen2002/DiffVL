import copy
import typing
from llm.pl.functool import get_ext
from llm.tiny import *
from .gene import Gene

mybool = dsl.bool
myint = dsl.int


class StageInfo:
    def   __init__(self, start, end, handler=None) -> None:
        self.start = start
        self.end = end
        self.handler = handler

    def __repr__(self) -> str:
        return f"{self.start}-{self.end}" + ("" if self.handler is None else f": {self.handler}")

    def no_grad(self):
        if self.handler is not None:
            assert self.start == self.end
            return True
        return False


class Trajectory:
    def __init__(self, duration, constraints: typing.List[Gene], start, stage_id, scene):
        self.start = start
        self.duration = duration
        self.constraints = constraints
        self.scene = scene
        self.stage_id = stage_id

    def add_gene(self, op, mode, name=None):
        gene = Gene(op, mode=mode, start=self.start, duration=self.duration, stage_id=self.stage_id, code=name)
        return Trajectory(self.duration, self.constraints + [gene], self.start, self.stage_id, self.scene)


trajectory = dsl.build_data_type(Trajectory)


def init_scene_if_not(scene):
    scene.cur_stage += 1
    if hasattr(scene, 'stage_timesteps') and len(scene.stage_timesteps) > 0:
        cur = scene.stage_timesteps[-1].end
    else:
        cur = 0
    return cur


list_float = List(myfloat)
@dsl.as_primitive
def __Scene__grasp(scene: scene_type, obj: soft_body_type, dir: list_float) -> scene_type:
    cur = init_scene_if_not(scene)
    from .libtool import MotionPlanner
    scene.stage_timesteps.append(
        StageInfo(cur, cur, MotionPlanner(scene.env, obj, dir))
    ) # duration zeros ..
    return scene

@dsl.as_primitive
def __Scene__cpdeform(scene: scene_type, obj: soft_body_type, target: PointCloudType, scale: list_float) -> scene_type:
    cur = init_scene_if_not(scene)
    from .cpdeform import CPDeform
    scene.stage_timesteps.append(
        StageInfo(cur, cur, CPDeform(scene.env, obj, target, scale=scale))
    ) # duration zeros ..
    return scene


@dsl.as_primitive
def __Scene__new_stage(scene: scene_type, duration:myint =1) -> trajectory:
    cur = init_scene_if_not(scene)
    scene.stage_timesteps.append(StageInfo(cur, cur + duration))
    return Trajectory(duration, [], scene.cur_step, scene.cur_stage, scene=scene)


@dsl.as_primitive
def __Trajectory__then(a: trajectory, b: trajectory) -> trajectory:
    duration = a.duration
    return Trajectory(duration + b.duration, a.constraints + [
        i.add_start_stage(a.duration) for i in b.constraints], start=a.start, stage_id=b.stage_id, scene=a.scene)


@dsl.as_primitive
def __Trajectory__together(a: trajectory, b: trajectory) -> trajectory:
    duration = a.duration
    raise NotImplementedError("Currently we do not support too complex temporal structures.")
    return Trajectory(duration + b.duration, a.constraints + b.constraints, start=a.start, scene=a.scene)


temporal_cond = Arrow(mybool)
import ast

def set_mode(a, b, mode):
    name = [ast.unparse(i) for i in get_ext().trace[-2].node.args]
    assert len(name) == len(b)
    for n, i in zip(name, b):
        a = a.add_gene(i, mode, n)
    return a

@dsl.as_primitive
def __Trajectory__sothat(a: trajectory, *b: VariableArgs(temporal_cond)) -> trajectory:
    # TODO: avoid execute the first term ..
    return set_mode(a, b, 'st')

@dsl.as_primitive
def __Trajectory__mean(a: trajectory, *b: VariableArgs(temporal_cond)) -> trajectory:
    return set_mode(a, b, 'mean')

@dsl.as_primitive
def __Trajectory__keep(a: trajectory, *b: VariableArgs(temporal_cond)) -> trajectory:
    return set_mode(a, b, 'keep')


@dsl.as_primitive
def __Trajectory__execute(a: trajectory) -> scene_type:
    scene = a.scene
    assert scene.cur_step == a.start, f"{scene.cur_step} {a.start}"
    stage_limit = get_ext().global_context['stage_limit']

    for i in a.constraints:
        if stage_limit is not None and i.stage_id >= stage_limit[0] and i.stage_id < stage_limit[1]:
            # print('stage limit', stage_limit, i.stage_id, i.start, i.duration)
            for out in i.execute(scene):
                if stage_limit is not None and (i.stage_id < stage_limit[1] - 1 or i.mode == 'keep') and i.mode != 'mean':
                    out['is_constrained'] = True
                else:
                    out['is_constrained'] = False
                scene.tables.append(out)

    scene.set_timestep(a.start + a.duration)
    return scene

    
def get_path(a):
    import os
    path = get_ext().global_context.get('path', None)
    if path is not None:
        a = os.path.join(path, a)
    return a


string = dsl.str
@dsl.as_primitive
def subgoal(a: string) -> scene_type:
    import torch
    from tools.utils import totensor
    state = torch.load(get_path(a))

    scene = Scene(None, False)
    scene.obs = [{'pos': totensor(state.X, 'cuda:0')}]
    scene.initial_state = state
    return scene


@dsl.as_primitive
def render_rgb(scene: tA, path: string) -> none:
    # render a scene or a soft body. please use this carefully .. 
    import cv2
    env = get_ext().global_context['env']
    if isinstance(scene, Scene):
        img = env.render_state_rgb(scene.initial_state)
    elif isinstance(scene, SoftBody):
        c = scene._color
        img = env.render_state_rgb(None, pos=scene.pcd().detach().cpu().numpy(), color=c)
    elif isinstance(scene, torch.Tensor):
        img = env.render_state_rgb(None, pos=scene.detach().cpu().numpy())
    else:
        raise NotImplementedError(f"{type(scene)}")
#     cv2.imwrite(get_path(path), img[..., ::-1])