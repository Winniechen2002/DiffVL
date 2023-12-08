import os
from omegaconf import OmegaConf
import glob
from diffsolver.launch.utils import sort_by_task

config_files = glob.glob("examples/single_stage_dev/*.yaml") + glob.glob("examples/single_stage_dev/*.yml")
tot = 0

tasks = []
for file in config_files:
    cfg = OmegaConf.load(file)

    path = sorted(glob.glob(os.path.join("examples/output/single_stage/debug_tool", cfg.saver.path, "*/tool_lang.txt")))
    if len(path) > 0:
        tasks.append(file.split('/')[-1])
        p = path[-1]
        if os.path.exists(p):
            print(tot, tasks[-1])
            tot += 1
            cfg.tool_sampler.code = str(open(p, "r").read().strip())
            # print(open(path, "r").read().strip())

            OmegaConf.save(cfg, file)
print(tasks, len(tasks))