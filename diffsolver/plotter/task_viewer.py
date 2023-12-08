# load and scan all the tasks
import os
from omegaconf import OmegaConf, DictConfig
from diffsolver.paths import get_path

TASKPATH = get_path('VISION_TASK_PATH') 

def collect_tasks():
    names = OmegaConf.load(os.path.join(TASKPATH, 'naming.yml'))
    assert isinstance(names, DictConfig)

    filenames = os.listdir(os.path.join(TASKPATH, 'static'))
    keys = [(int(i.split('_')[-1]), idx) for idx, i in enumerate(filenames) if os.path.isdir(os.path.join(TASKPATH, 'static', i))]


    keys = sorted(keys)
    filenames = [filenames[i[1]] for i in keys]

    images = []

    cur = os.getcwd()

    for i in filenames:
        item = {}
        path = os.path.join(TASKPATH, 'static', i)

        item['name'] = i

        out = []
        for p in os.listdir(path):
            if 'obj' not in p:
                out.append(os.path.relpath(os.path.abspath(os.path.join(TASKPATH, 'static', i, p)), cur))
        item['images'] = sorted(out)

        obj_path = os.path.join(TASKPATH, 'objs', i.split('_')[-1] + '.png')
        if os.path.exists(obj_path):
            item['objs'] = os.path.relpath(os.path.abspath(obj_path), cur)


        index = int(i.split('_')[-1])
        if index in names:
            item['item_names'] = names[index]
        images.append(item)

    return images


if __name__ == '__main__':
    collect_tasks()
