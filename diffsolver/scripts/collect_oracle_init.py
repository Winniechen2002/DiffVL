import os

FOLDER = 'dev'
# print(os.listdir('examples/single_stage_dev'))

for i in os.listdir(f'examples/single_stage_{FOLDER}'):
    if not i.startswith('task18'):
        continue
    os.system(f"CUDA_VISIBLE_DEVICES=0 python3 run_single_stage.py oracle --config examples/single_stage_{FOLDER}/{i} tool_sampler.n_samples=10000 run_solver=False saver.path=tmp/oracle_tool_init/{i}")
    try:
        with open(f"tmp/oracle_tool_init/{i}/qpos.txt", 'r') as f:
            print(i, f.read())
    except FileNotFoundError:
        pass