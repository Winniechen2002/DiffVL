from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Union
from tools.config import CN


def parse_dict(inp) -> Union[CN, list, tuple]:
    if isinstance(inp, list):
        out = []
        for i in inp:
            out.append(parse_dict(i))
        return out
    elif isinstance(inp, dict):
        out = CN(new_allowed=True)
        for k, v in inp.items():
            out[k] = parse_dict(v)
        return out
    try:
        x = eval(inp)
        return x
    except Exception as e:
        pass

    return inp

def list2tuple(inp):
    if isinstance(inp, list) or isinstance(inp, tuple) or isinstance(inp, ListConfig):
        out = []
        for i in inp:
            out.append(list2tuple(i))
        return tuple(out)
    elif isinstance(inp, dict) or isinstance(inp, CN) or isinstance(inp, DictConfig):
        out = {}
        for k, v in inp.items():
            out[k] = list2tuple(v)
        return out
    return inp

def CN2dict(inp, list2tuple=False):
    if isinstance(inp, list) or isinstance(inp, tuple):
        out = []
        for i in inp:
            out.append(CN2dict(i))
        if list2tuple or isinstance(inp, tuple):
            out = tuple(out)
        return out
    elif isinstance(inp, dict):
        out = {}
        for k, v in inp.items():
            out[k] = CN2dict(v)
        return out
    elif isinstance(inp, CN):
        out = {}
        for k, v in inp.items():
            out[k] = CN2dict(v)
        return out
    return inp

    
def dict2CN(kwargs):
    #dict = parse_dict(CN._load_cfg_from_file(open(cfg_path, 'r')))
    kwargs = OmegaConf.create(kwargs)
    kwargs = OmegaConf.to_yaml(kwargs)
    return CN._load_cfg_from_yaml_str(kwargs)