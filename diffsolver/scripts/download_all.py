#os.system("kubectl cp hza-pod:/cephfs/hza/diffsolver/models/single_stage/ ./examples/output/single_stage/")
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('modes', type=str, default=None)
parser.add_argument('tasks', type=str, default=None)
args = parser.parse_args()

for mode in args.modes.split(','):
    if args.tasks != 'all':
        for task in args.tasks.split(','):
            os.system("sync.py --path examples/output/single_stage/{}/{} --run".format(mode, task))
    else:
        os.system("sync.py --path examples/output/single_stage/{} --run".format(mode))