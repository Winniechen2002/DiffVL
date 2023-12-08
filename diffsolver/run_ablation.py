import argparse
from typing import List, Tuple
import os
from diffsolver.launch.utils import keys, sort_by_task



def main():
    methods = ['oracle', 'lang', 'sac', 'ppo', 'emdonly', 'badinit', 'oracle_lang', 'oracle_emdonly', 'debug_tool']

    TASKPATH = os.path.join(os.path.dirname(__file__), 'examples/single_stage_dev/')

    tasks = sort_by_task(os.listdir(TASKPATH))

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='shape of the domain', choices=methods)
    parser.add_argument('--task', type=str, choices=tasks)
    parser.add_argument('--start', action='store_true', help='run the command')
    args, unknown = parser.parse_known_args()


    if not args.method:
        args.method = keys('Select methods', methods)

    
    if not args.task:
        args.task = keys('Select tasks', tasks)

    cmdline = f"python -m diffsolver.run_single_stage {args.method} --config {TASKPATH}/{args.task} " + ' '.join(unknown)
    print(cmdline)

    if not args.start:
        yes_no = keys('Run?', ['yes', 'no'])
        if yes_no == 'no':
            return

    os.system(cmdline)
    


if __name__ == '__main__':
    main()
