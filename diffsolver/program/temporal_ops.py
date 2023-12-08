import torch
import warnings
from .types.types import *
from .types import SceneSpec
from .types.constraints import AndConstraint, LastConstraint, KeepConstraint
from .prog_registry import register_prog
from typing import Sequence


def get_weight_from_loss(loss, weight: float|None=None):

    if weight is None:
        if hasattr(loss, 'default_weight') and loss.default_weight is not None:
            _weight = loss.default_weight 
        else:
            _weight = 1.
    else:
        _weight = weight
    return _weight

@register_prog('tlast')
def tlast(observation_loss: STEP_COND, weight: float|None=None, _end=1., strict=False) -> SCENE_COND:
    warnings.warn("tlast is deprecated, use last instead")
    def _last(x: SceneSpec):
        x.set_timestep(int((x.total_steps() - 1) * _end))
        _weight = get_weight_from_loss(observation_loss, weight) * 20
        # print(observation_loss, _weight)


        return LastConstraint(
            *observation_loss(x), code="last({}, {}, {})".format(str(observation_loss), _end, strict), 
            sync=observation_loss.synchronize(), weight=_weight, is_constraint=strict
        )
    return _last


@register_prog('tkeep')
def tkeep(observation_loss: STEP_COND, weight: float|None, _start=0., _end=1., strict=False) -> SCENE_COND:
    warnings.warn("tkeep is deprecated, use keep instead")
    def _keep(x: SceneSpec):
        sat = True
        loss = 0
        _weight = get_weight_from_loss(observation_loss, weight)
        # print(observation_loss, _weight, observation_loss.default_weight)
        for t in range(int(_start * x.total_steps()), int(_end * x.total_steps())): 
            x.set_timestep(t)
            try:
                _loss, _sat = observation_loss(x)
            except Exception as e:
                print("Error in tkeep", e, str(observation_loss))
                raise e
            sat = sat and _sat
            loss += _loss
        assert isinstance(loss, torch.Tensor)
        assert isinstance(sat, torch.Tensor)
        return KeepConstraint(
            loss, sat, code="keep({}, {}, {}, {})".format(str(observation_loss), _start, _end, strict), 
            sync=observation_loss.synchronize(), weight=_weight, is_constraint=strict
        )
    return _keep



@register_prog('last')
def last(observation_loss: STEP_COND, end=1., strict=False, weight: float|None=None) -> SCENE_COND:
    def _last(x: SceneSpec):
        x.set_timestep(int((x.total_steps() - 1) * end))
        _weight = get_weight_from_loss(observation_loss, weight) * 20

        return LastConstraint(
            *observation_loss(x), code="last({}, {}, {})".format(str(observation_loss), end, strict), 
            sync=observation_loss.synchronize(), weight=_weight, is_constraint=strict
        )
    return _last


@register_prog('keep')
def keep(observation_loss: STEP_COND, start=0., end=1., strict=False, weight: float|None=None) -> SCENE_COND:
    def _keep(x: SceneSpec):
        sat = True
        loss = 0
        _weight = get_weight_from_loss(observation_loss, weight) * 20
        # print(observation_loss, _weight, observation_loss.default_weight)
        for t in range(int(start * x.total_steps()), int(end * x.total_steps())): 
            x.set_timestep(t)
            _loss, _sat = observation_loss(x)
            sat = sat and _sat
            loss += _loss
        assert isinstance(loss, torch.Tensor)
        assert isinstance(sat, torch.Tensor)
        return KeepConstraint(
            loss, sat, code="keep({}, {}, {}, {})".format(str(observation_loss), start, end, strict), 
            sync=observation_loss.synchronize(), weight=_weight, is_constraint=strict
        )
    return _keep


@register_prog('tand')
def tand(*args: SCENE_COND) -> SCENE_COND:
    def _tand(x: SceneSpec):
        return AndConstraint(*[f(x) for f in args])
    return _tand

    
def multi_stages(stages: Sequence[SCENE_COND], horizons: Sequence[int]):
    def _eval_multi_stages(x: SceneSpec):
        original_obs = x.obs

        start = 0
        outs = []
        for stage, horizon in zip(stages, horizons):
            x.obs = original_obs[start: start + horizon]
            outs.append(stage(x))
            start += horizon
        return AndConstraint(*outs)
    return _eval_multi_stages
