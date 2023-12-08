# deform into an L shape

from llm.genetic import *

def makeL(scene: Scene):
    """
    scene: L.yml
    env_cfg:
        sim_cfg:
            ground_friction: 5.
    """

    obj = scene.obj(0)

    tool = scene.tool()
    L = concat_pcd(obj.left().pcd(), obj.right().front().pcd())
    deform = lambda: emd(obj.pcd(), L) * 10. < 0.0

    contact = contact_obj_fn(tool, obj)

    def fn():
        scene.cpdeform(obj, L, [0.2, 0., 0.2, 0., 1., 0.])
        scene.new_stage(20).sothat(contact).mean(deform).execute()
        stable_release(scene)

    loop(fn, 3)
    return scene