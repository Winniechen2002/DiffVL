import os
import glob

# Get all yaml/yml files in the examples/single_stage directory
#config_files = glob.glob("examples/single_stage_dev/*.yaml") + glob.glob("examples/single_stage_dev/*.yml")

tasks = """
Task1
Task10
Task13
Task32
Task40
Task42
Task43
Task45
Task47
Task48
Task52
Task54
Task56
Task57
Task59
Task60
Task62
Task70
Task72
Task73
"""

tasks = tasks.split('\n')
tasks = [t for t in tasks if t != '']

tasks = ['t' + t[1:] for t in tasks]
print(tasks)
algo = 'ppo'
for t in tasks:
    p = 'examples/multistage/' + t + '/total.yaml'
    if not os.path.exists(p):
        p = 'examples/multistage/' + t + '/total.yml'

    job_name = t + '-multi-' + algo
    #if job_name in ['task70-cut']:
    command = f"remote_run.py run_multistage.py {algo} --config {p} --run --job_name {job_name}"
    os.system(command)
        # print(command)