#!/usr/bin/env python3

import gc
import cv2
import matplotlib.pyplot as plt
from diffsolver.visiontask import TaskSeq
from diffsolver.utils import MultiToolEnv

def test():
    env = MultiToolEnv()

    for i in range(31):
        try:
            task = TaskSeq(i+1)
            image = task.render_state_images(env)
        except ValueError:
            continue

        cv2.imwrite(f'test{i+1}.png', image[..., ::-1])

if __name__ == '__main__':
    test()