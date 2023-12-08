import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, default=None)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--modes', type=str, default='badinit,emdonly,lang')
args = parser.parse_args()

#name = sys.argv[1]

for i in range(args.seed):
    for mode in args.modes.split(','):
        command = f"python3 run_single_stage.py {mode} --config examples/single_stage_dev/{args.name}"
        os.system(command)