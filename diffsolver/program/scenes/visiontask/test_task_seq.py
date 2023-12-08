# %%
def test():
    import gc
    import cv2
    import matplotlib.pyplot as plt
    from diffsolver.program.scenes.visiontask import TaskSeq

    task = TaskSeq(8)
    print(f"{task.naming=}")

    plt.imshow(task.fetch_scene_images())
    plt.savefig('test0.png')

    state, objects = task.fetch_stage(0)
    for k, v in objects.items():
        print(f"{k=}, {len(v)=}")

    #print(scene_config.names)
    #scene_config.views
    import numpy as np
    print(f"{np.unique(state.ids)=}")

    from diffsolver.utils import MultiToolEnv
    env = MultiToolEnv()
    cv2.imwrite('objects.png', task.render_stage_objects(env, 0)[..., ::-1])

    del env 
    gc.collect()