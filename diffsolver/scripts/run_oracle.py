import os

print(os.listdir('examples/single_stage_dev'))

#method = 'emdonly'
#method = 'oracle_lang'
#method = 'oracle_emdonly'
# method = 'badinit'
method = 'sac'

for i in os.listdir('examples/single_stage_dev'):
    # os.system(f"CUDA_VISIBLE_DEVICES=1 python3 run_ablation.py --method oracle --task {i} tool_sampler.n_samples=10000 run_solver=False saver.path=tmp/oracle_tool_init/{i} --run")
    p = i.split('.')[0]
    if p in ['task10_stage1', 'task22_stage1', 'task24_carving', 'task25_pick_and_palce', 'task28_wrap', 'task2_wind', 'task33_pick_place', 'task38_deform']: #, 'task11_wind', 'task13_pick_place', 'task39_cut']:
        # print(i)
        task_name = (p + '-' + method).replace('_', '-')
        #os.system(f"remote_run.py run_ablation.py --method {method} --task {i} --start --run --job_name {task_name}")
        print(f"remote_run.py run_ablation.py --method {method} --task {i} --start --run --job_name {task_name}")