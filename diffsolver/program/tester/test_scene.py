import gc
from diffsolver.utils.test_utils import generate_scene_traj


def test_scene():
    env, scene = generate_scene_traj()
    assert scene.total_steps() == 31
    assert scene.cur_step == 0

    objects = scene.get_object_list()
    assert len(objects) == 2

    assert objects[0].N() == 1964
    assert objects[1].N() == 200
    objects[0].surface()

    tool = scene.tool()
    assert tool.qpos().shape == (7,) 
    # assert dist2body(tool, objects[0]) > dist2body(tool, objects[1])

    del objects, tool, env, scene
    gc.collect()