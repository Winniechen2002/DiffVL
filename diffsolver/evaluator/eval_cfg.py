from typing import List
from dataclasses import dataclass, field


@dataclass
class PluginConfig:
    pass


@dataclass
class IOUConfig(PluginConfig):
    success: float = 0.8

class SaveSceneConfig(PluginConfig):
    pass

class CodelineConfig(PluginConfig):
    pass


@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda : ['iou'])
    iou: IOUConfig = field(default_factory=IOUConfig)
    save_scene: SaveSceneConfig = field(default_factory=SaveSceneConfig)
    code: CodelineConfig = field(default_factory=CodelineConfig) 