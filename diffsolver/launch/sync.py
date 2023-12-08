#!/usr/bin/env python3
import os
import sys
from diffsolver.launch.utils import get_sync_cmd, keys


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()

    if args.path is None:
        args.path = os.getcwd()
    cmd = get_sync_cmd(args.path)
    print(cmd)
    
    if not args.run:
        run = keys('start?', ['yes', 'no'])
    else:
        run = 'yes'
    if run == 'yes':
        os.system(cmd) 

    


if __name__ == '__main__':
    main()
