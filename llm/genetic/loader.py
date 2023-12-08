import numpy as np
from llm.tiny import dsl
from envs import WorldState
from tools import CN


def parse_dict(inp):
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

def load_scene(cfg_path):
    # represent scene with yaml.. 
    dict = parse_dict(CN._load_cfg_from_file(open(cfg_path, 'r')))

    import torch
    if 'Path' in dict:
        state = torch.load(dict['Path'])
    else:
        state = WorldState.sample_shapes(**dict['Shape'])
    if "Tool" in dict:
        state = state.switch_tools(**dict["Tool"])
    if "Worm" in dict:
        state.worm_scale = dict["Worm"]["action_scale"]
    return state

import ast

def load_prog(path):
    with open(path, 'r') as f:
        source = '\n'.join([i for i in f.readlines()])

    tree = ast.parse(source)
    assert isinstance(tree, ast.AST)
    
    funcs = []
    for body in tree.body:
        if isinstance(body, ast.ImportFrom):
            assert body.names[0].name == '*'
            import importlib
            print('importing', body.module, '..')
            importlib.import_module(body.module)
        elif isinstance(body, ast.FunctionDef):
            docstring = ast.get_docstring(body, clean=True)
            if docstring is not None:
                body.body = body.body[1:]
            funcs.append([ast.unparse(body), docstring])
        else:
            raise NotImplementedError(f"can't parse {type(body)} of \n{ast.unparse(body)}")

    out = []
    for f, doc in funcs[:-1]:
        out.append(dsl.as_func(f))
        out[-1].doc = doc
        out[-1].source = f

    f = dsl.parse(funcs[-1][0])
    f.doc = funcs[-1][1]
    f.source = funcs[-1][0]
    return f, out