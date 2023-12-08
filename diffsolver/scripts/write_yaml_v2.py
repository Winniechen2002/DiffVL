import os
import yaml
from yaml import Dumper
from omegaconf import OmegaConf
import glob
from diffsolver.launch.utils import sort_by_task

def custom_str_representer(dumper, data):
    style = '|' if '\n' in data else None
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=style)

yaml.add_representer(str, custom_str_representer)



config_files = glob.glob("examples/single_stage_dev/*.yaml") + glob.glob("examples/single_stage_dev/*.yml")
tot = 0

tasks = []
for file in config_files:
    cfg = OmegaConf.load(file)

    path = sorted(glob.glob(os.path.join("examples/output/single_stage/debug_phys", cfg.saver.path, "*/new_config.yaml")))
    if len(path) > 0:
        tasks.append(file.split('/')[-1])
        p = path[-1]
        if os.path.exists(p):
            print(tot, tasks[-1])
            t = OmegaConf.load(p)
            print(t.prog.code)
            tot += 1
            print(t.prog.translator.code)


            cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(prog=dict(translator=dict(code=str(t.prog.code))))))
            #print(cfg.prog.translator.code)

            container = OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
            out = yaml.dump(  # type: ignore
                container,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )

            print(out)
            with open(file, 'w') as f:
                f.write(out)
            # print(open(path, "r").read().strip())

            # OmegaConf.save(cfg, file)
print(tasks, len(tasks))