import os
import pickle
from ..paths import get_path


def hash_object(obj):
    import json
    import hashlib
    json_str = json.dumps(obj, sort_keys=True)
    hash_result = hashlib.sha256(json_str.encode('utf-8')).digest()
    return hex(int.from_bytes(hash_result, byteorder='big'))[2:]



def load_scene(name, clear_cache=False, _cache_dir=None, **kwargs):
    from ..program.scenes import load_scene_with_envs
    from ..config import SceneConfig
    config = SceneConfig(path=name, **kwargs)

    
    _cache_dir = _cache_dir or get_path('CACHE_DIR')
    os.makedirs(_cache_dir, exist_ok=True)


    hashpath = os.path.join(_cache_dir, hash_object((name, str(kwargs))) + '.pkl')

    if os.path.exists(hashpath) and not clear_cache:
        with open(hashpath, 'rb') as f:
            return pickle.load(f)

    out = load_scene_with_envs(None, config)
    with open(hashpath, 'wb') as f:
        pickle.dump(out, f)

    return out