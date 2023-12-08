from typing import TypeVar, Union
from omegaconf import DictConfig
T = TypeVar('T')
ConfigType = Union[T, DictConfig]


from .io import *
from .misc import *
from .test_utils import *
from ..paths import get_path
from .rendering import rendering_objects

print('What is the cuda visible devices?', os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET'))