from .evaluator.eval_cfg import EvalConfig
from dataclasses import dataclass, field
from typing import List, Any, Dict


@dataclass
class OptimConfig:
    lr: float = 0.05
    enable_constraints: bool = False
    frank_wolfe: Any = field(default_factory=lambda: {})


@dataclass
class TrainerConfig:
    nsteps: int = 100


@dataclass
class Logger:
    format_strs: List[str] = field(default_factory=lambda: ["log", "csv"])
    date: bool = False


@dataclass
class CameraConfig:
    pos: List[float] = field(default_factory=lambda: [0.5, 0.5, 2.])
    quat: List[float] = field(default_factory=lambda: [0.5, -0.5, 0.5, 0.5])

@dataclass
class LookatConfig:
    center: List[float] = field(default_factory=lambda: [0.5, 0.5, 0])
    theta: float = 0.
    phi: float = 3.14/4
    zeta: float = 0.
    radius: float = 3.


@dataclass
class SaverConfig:
    path: str = '' # path to the assets folder
    logger: Logger = Logger()
    render_interval: int = 50

    use_wandb: bool = False
    wandb_group: str|None = None
    wandb_name: str|None = None

    render_mode: str = 'rgb_array'
    use_lookat: bool = True
    camera_config: CameraConfig = CameraConfig()
    lookat_config: LookatConfig = LookatConfig()


    
@dataclass
class SceneConfig:
    path: str = 'block.yml' # something located in diffsolver/assets/; or string like 10_3.task
    Tool: Any = field(default_factory=lambda: {})
    Shape: Any = field(default_factory=lambda: {})
    Objects: Any = field(default_factory=lambda: {})
    rename: Dict[str, str] = field(default_factory=lambda: {})

    # configs for loading past stages
    use_config: bool = False
    state_id: int = -1

    goal: str = 'none'  # could also be something like 10_4.task if necessary
    

@dataclass
class MPConfig:
    """
    config for motion planning
    """
    expand_dis: float = 0.1
    step_size: float = 0.02
    max_iter: int = 0
    use_lvc: bool = False
    info: bool = True
    tolerance: float = 0.0 # tolerance for collision checking


@dataclass
class ToolSamplerConfig:
    sampler: str  = 'default'
    use_tips: bool = False

    n_samples: int = 1000
    optimize_loss: bool = False

    n_sol: int = 1

    equations: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    motion_planner: MPConfig = MPConfig()


    lang: str = ''
    use_lang: bool = False # used for translator

    use_code: bool = False
    code: str|None = None # store the code generated by lang

    use_default: bool = False

    use_cpdeform: bool = False




@dataclass
class TranslatorConfig:
    version: str = 'v1'
    explanation: bool = True
    verify_scene: bool = True
    code: str|None = ''   # used for precode



@dataclass
class DiffPhysStageConfig:
    code: str
    horizon: int
    stages: List[Any] = field(default_factory=list)

    lang: str = ''
    max_retry: int = 0
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)


@dataclass
class DefaultConfig:
    max_steps: int = 1024
    sample: int = 2000
    seed: int|None = None

    run_solver: bool = True


    scene: SceneConfig = SceneConfig() 

    optim: OptimConfig = OptimConfig() 
    trainer: TrainerConfig = TrainerConfig()
    saver: SaverConfig = SaverConfig()
    
    tool_sampler: ToolSamplerConfig = ToolSamplerConfig(n_samples=0)

    prog: DiffPhysStageConfig = DiffPhysStageConfig(horizon=20, code='lift()')
    stages: List[DiffPhysStageConfig] = field(default_factory=list)

    evaluator: EvalConfig = EvalConfig()

    substages: Any = None