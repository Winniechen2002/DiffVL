#!/usr/bin/env python
# remote_run.py --job_name xxxx main.py xxkjklfjkwejiowejoi
import os
import tempfile
import datetime
import argparse
from enum import Enum
from typing import List, Dict, cast
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig

FILEPATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = OmegaConf.load(os.path.join(FILEPATH, 'job_template.yml'))
assert isinstance(TEMPLATE, DictConfig)




@dataclass
class Config:
    _MACROS: Dict[str, str] = field(default_factory=dict)
    _CMDS: List[str] = field(default_factory=list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch_config', type=str, default='launch.yml')
    parser.add_argument('--job_name', type=str, default='job', required=True)
    parser.add_argument('--not_git', action='store_true')
    parser.add_argument('--run', action='store_true')
    args, unknown = parser.parse_known_args()

    cmd = 'python ' + ' '.join(unknown)
    print(cmd)

    if not args.not_git:
        os.system("git commit -am 'update'")
        os.system("git push")

    dict_cfg:DictConfig = OmegaConf.structured(Config)
    dict_cfg.merge_with(OmegaConf.load(os.path.join(FILEPATH, args.launch_config)))

    OmegaConf.resolve(dict_cfg)
    cfg = cast(Config, dict_cfg)

    macros = '; '.join([f'export {k}={v}' for k, v in cfg._MACROS.items()])
    cmds = '; '.join(cfg._CMDS + [cmd])


    workspace = os.path.join('/root', os.path.relpath(os.getcwd(), os.path.join(FILEPATH, '../../../')))


    val: str = TEMPLATE['spec']['template']['spec']['containers'][0]['args'][0]
    val = val.replace('$MACROS', macros).replace('$CMD', cmds).replace('$WORKSPACE', workspace)

    TEMPLATE['metadata']['name'] = 'hza-job-'+args.job_name
    TEMPLATE['spec']['template']['spec']['containers'][0]['args'][0] = val

    print(OmegaConf.to_yaml(TEMPLATE))


    temp_path = os.path.join(tempfile.gettempdir(), 'diffsolver')
    os.makedirs(temp_path, exist_ok=True)
    
    save_path = os.path.join(
        temp_path,
        datetime.datetime.now().strftime(f"{args.job_name}%Y%m%d%H%M%S%f"),
    )
    print(f'Save to {save_path}')
    with open(save_path, 'w') as f:
        f.write(OmegaConf.to_yaml(TEMPLATE))

    if args.run:
        os.system(f'kubectl create -f {save_path}')
    
    

    
if __name__ == '__main__':
    main()