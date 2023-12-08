from typing import Dict, List
import enum
from omegaconf import OmegaConf
import numpy as np
import os
from numpy.typing import NDArray

class Curve:
    def __init__(self, mean, std=None, xs: NDArray|None=None) -> None:
        self.xs = xs  if xs is not None else np.arange(len(mean))
        self.mean: NDArray[np.float32] = mean
        self.stds: NDArray[np.float32] = np.zeros_like(mean) if std is None else std

class ExperimentOutput:
    def __init__(self) -> None:
        self.labels: Dict[str, str] = {}
        self.scalars: Dict[str, float] = {}
        self.videos: Dict[str, str] = {}
        self.images: Dict[str, str] = {} # image path
        self.texts: Dict[str, str] = {}
        self.curves: Dict[str, Curve] = {} # curve path

    def __str__(self) -> str:
        return OmegaConf.to_yaml(OmegaConf.create(self.__dict__))



class Task(enum.Enum):
    tool_sample="tool_sample"
    phys_sample="phys_sample"
    solve_tool="solve_tool"
    solve_dev="solve_dev" # solve the progs and tasks in examples/single_stage_dev
    multi_dev="multi_dev" # plot results in multistage_dev
    multi_prog="multi_prog" # plot results in multistage
    multi_long="multi_long" # plot results in multistage
    multi_short="multi_short" # plot results in multistage
    multi_mid="multi_mid" # plot results in multistage
    solve_lang="solve_lang" # solve_lang_dev for single stage
    ablation="ablation" # run ablation study
    ablation_multi="ablation_multi" # run ablation study for multistage
    ablation_multi_singleonly="ablation_multi_singleonly" # run ablation study for multistage and single
