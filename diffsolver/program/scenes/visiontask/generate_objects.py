import gc
import os
import cv2
import matplotlib.pyplot as plt
from diffsolver.program.scenes.visiontask import TaskSeq
from diffsolver.paths import get_path

from diffsolver.utils import MultiToolEnv
env = MultiToolEnv(sim_cfg=dict(max_steps=10)) # type: ignore

TAKS_PATH = os.path.join(get_path('VISION_TASK_PATH'), 'objs')
os.makedirs(TAKS_PATH, exist_ok=True)

#for i in range(1000):
for i in [5]:
    try:
        task = TaskSeq(i)
    except Exception:
        continue

    print(i)
    # print(f"{task.naming=}")

    # print(f"{task.db=}")
    # plt.imshow(task.fetch_scene_images())
    # plt.savefig('test0.png')

    for s in range(task.num_stages):
        suffix = '' if s == 0 else f'_{s}'
        cv2.imwrite(os.path.join(TAKS_PATH, f'{i}{suffix}.png'), task.render_stage_objects(env, s)[..., ::-1])