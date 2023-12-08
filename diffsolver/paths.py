"""
Define paths for data and assets
One can override the default paths by setting the environment variables
and then call `get_path` to get the path.
"""
import os
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(FILE_PATH,  'data')
ASSET_DIR = os.path.join(FILE_PATH, 'assets')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

DICTS = {
    'DATA_DIR': DATA_DIR,
    'ASSET_DIR': ASSET_DIR,
    'MODEL_DIR': MODEL_DIR,
}

def get_path(name, *args):
    if name in os.environ:
        path =  os.environ[name]
    else:
        path = DICTS[name]
    if len(args) > 0:
        path = os.path.join(path, *args)
    return path

def touch(*args, makedir=False):
    path = os.path.join(*args)
    if makedir:
        os.makedirs(path, exist_ok=True)
    return path

DICTS['CACHE_DIR'] = touch(get_path('DATA_DIR'), 'cache', makedir=True)
DICTS['INIT_CFG_DIR'] = touch(get_path('ASSET_DIR'), 'init_cfg')
DICTS['VISION_TASK_PATH'] = touch(get_path('ASSET_DIR'), 'TASK', makedir=False)

__all__ = ['get_path', 'touch']