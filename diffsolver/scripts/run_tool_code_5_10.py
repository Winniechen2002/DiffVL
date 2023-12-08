import os
from diffsolver.launch.utils import sort_by_task 
tasks = sort_by_task(['task39_cut.yaml', 'task25_pick_place.yaml', 'task57_1.yaml', 'task37_cut.yaml', 'task22_stage1.yaml', 'task38_deform.yaml', 'task33_pick_place.yaml', 'task11_wind.yaml', 'task50_3.yaml', 'task2_wind.yaml', 'task62_2.yaml', 'task65_1.yaml', 'task18.yaml', 'task28_wrap.yaml', 'task13_pick_place.yml', 'task10_stage1.yml', 'task24_carving.yml', 'task7_press.yml'])
print(tasks)

# exit(0)

for i in tasks:
    # os.system(f"CUDA_VISIBLE_DEVICES=1 python3 run_ablation.py --method oracle --task {i} tool_sampler.n_samples=10000 run_solver=False saver.path=tmp/oracle_tool_init/{i} --run")
    # if '57' not in i and '65' not in i:
    #     continue
    p = i.split('.')[0]
    task_name = (p + '-' + 'debug-emd').replace('_', '-')
    #os.system(f"remote_run.py run_ablation.py --method {method} --task {i} --start --run --job_name {task_name}")
    #cmd = f"remote_run.py run_single_stage.py debug --config examples/single_stage_dev/{i} --run --job_name {task_name}"
    cmd = f"remote_run.py run_single_stage.py debug --config examples/single_stage_dev/{i} --run --job_name {task_name}"
    #cmd = f"python3 run_single_stage.py debug --config examples/single_stage_dev/{i}"
    os.system(cmd)
    print(cmd)