from enum import Enum
from omegaconf import DictConfig
from dataclasses import dataclass, field
from diffsolver.config import DefaultConfig
from typing import Any, List


BaseConfigType = DefaultConfig | DictConfig

class Tasks(Enum):
    run = 0
    render = 1


@dataclass
class SubConfig:
    name: str | None = None
    config: str | None = None
    modifier: Any = field(default_factory=lambda: {})

    
@dataclass
class RenderConfig:
    port: int = 8000                                    # port for the renderer

    mode: str = 'all'                                   # all, finihsed, time, failed


@dataclass
class ManagerConfig:
    path: str                                           # path to store all runs
    base_path: str | None = None                        # path to the base config
    task: Tasks = Tasks.render                          # for render task we do not run
    debug_run: bool = False                             # if True, we will not run the config
    max_retry: int = 3

    subsequent: bool = False

    common: Any = field(default_factory=lambda: {})     # modifier over the base config

    seeds: None | list[int] = None                      # if a list of seeds is provided, we will run the same config for multiple times
    sweep_mode: str = 'list'                            # the way to sweep the variations, currently only support list

    variations: List[SubConfig] = field(default_factory=list)   # a list of variations to sweep over the base config, used when sweep_mode is list
    subset: None | list[str] = None                     # if a list of names is provided, we will only run the configs with the names in the list

    renderer: RenderConfig = field(default_factory=RenderConfig)