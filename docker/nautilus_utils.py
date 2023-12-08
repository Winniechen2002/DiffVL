#!/usr/bin/env python3
import os
import subprocess

while True:
    out = subprocess.check_output('kubectl get pods'.split(' ')).decode('utf-8').strip().split('\n')

    started = None
    for i in out:
        if 'hza-task-annotate' in i:
            started = 'Running' in i
            break
    if started is None:
        os.system(f"kubectl create -f pod.yml")
    elif started:
        break

def work(port):
    os.system(f'kubectl port-forward hza-task-annotate {port}:{port}')


import multiprocessing as mp

p1 = mp.Process(target=work, args=(5000,))
p2 = mp.Process(target=work, args=(5001,))
p1.start(); p2.start()
p1.join(); p2.join()