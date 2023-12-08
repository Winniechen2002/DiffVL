import os
import llm
import pickle
from llm.genetic.loader import load_scene

cache_dir = os.path.join(llm.CACHEPATH, 'libstate')
PATH = os.path.dirname(os.path.abspath(__file__))

def str2state(name, clear_cache=False, _cache_dir=None):
    _cache_dir = _cache_dir or cache_dir
    os.makedirs(_cache_dir, exist_ok=True)

    path = os.path.join(_cache_dir, name + '.pkl')

    if os.path.exists(path) and not clear_cache:
        with open(path, 'rb') as f:
            return pickle.load(f)

        
    if name.endswith('.yml'):
        state = load_scene(os.path.join(PATH, 'dataset', 'config', name))
    else:
        raise NotImplementedError

    with open(path, 'wb') as f:
        pickle.dump(state, f)

    return state