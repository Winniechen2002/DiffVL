import os
import sys
from diffsolver.launch.utils import sort_by_task

#tasks = sort_by_task(['task39_cut.yaml', 'task25_pick_place.yaml', 'task57_1.yaml', 'task37_cut.yaml', 'task22_stage1.yaml', 'task38_deform.yaml', 'task33_pick_place.yaml', 'task11_wind.yaml', 'task50_3.yaml', 'task2_wind.yaml', 'task62_2.yaml', 'task65_1.yaml', 'task18.yaml', 'task28_wrap.yaml', 'task13_pick_place.yml', 'task10_stage1.yml', 'task24_carving.yml', 'task7_press.yml', 'task36_rope.yaml'])
#tasks = ['task5_moveover.yml']

tasks = """
task7_press.yml
task38_deform.yaml
task11_wind.yaml
task24_carving.yml
task10_stage1.yml
task22_stage1.yaml
task5_moveover.yml
task13_pick_place.yml
task2_wind.yaml
task36_rope.yaml
task57_1.yaml
task50_3.yaml
task18.yaml
task65_1.yaml
task62_2.yaml
task28_wrap.yaml
task37_cut.yaml
task39_cut.yaml
task70_cut.yaml
task25_pick_place.yaml
"""

# tasks = """
# task7_press.yml
# task11_wind.yaml
# task5_moveover.yml
# task37_cut.yaml
# task50_3.yaml
# task70_cut.yaml
# """
tasks = tasks.strip().split('\n')
tasks = sort_by_task(tasks)
#tasks = ['task70_cut.yaml', 'task37_cut.yaml']
for i in tasks:
    assert os.path.exists(f'examples/single_stage_dev/{i}')
#print('\n'.join(tasks))
#exit(0)


for i in tasks:
    for seed in [0]:
        # os.system(f"CUDA_VISIBLE_DEVICES=1 python3 run_ablation.py --method oracle --task {i} tool_sampler.n_samples=10000 run_solver=False saver.path=tmp/oracle_tool_init/{i} --run")
        # if '57' not in i and '65' not in i:
        #     continue
        p = i.split('.')[0]
        task_name = (p + '-' + 'ppo').replace('_', '-')
        if seed > 0:
            task_name += f'-{seed}'
        cmd = f"remote_run.py run_single_stage.py ppo --config examples/single_stage_dev/{i} --run --job_name {task_name}-2"
        os.system(cmd)
        print(cmd)