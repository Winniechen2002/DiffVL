from llm.genetic import *

def fling(scene: Scene):
    """
    scene: rope_hang.yml
    env_cfg:
        sim_cfg:
            ground_friction: 0.1
    """
    rod = scene.obj(0)
    rope = scene.obj(1)
    rope_left_end = rope.leftmost(0.05)
    rope_right_end = rope.rightmost(0.05)


    contact = contact_obj_fn(scene.tool(), rope_left_end)
    rod_initial = rod.pcd()
    fix = lambda: chamfer(rod.pcd(), rod_initial) < 0.0002

    def liftit(obj: SoftBody):
        return lambda: min_array(obj.pcd().y()) > max_array(rod.pcd().y())
    lift_left = liftit(rope.left())

    dist = lambda: (rope_right_end.pcd().com() - rod.upmost(0.05).pcd().com()).norm()
    touch = lambda: dist() < 0.05
    beright = lambda: min_array(rope_right_end.pcd().x()) > max_array(rope_left_end.pcd().x())

    scene.grasp(rope_left_end, [0., 1., 0.])


    scene.new_stage(20).sothat(contact).then(
        scene.new_stage(40).keep(contact, fix).sothat(lift_left)
    ).then(
        scene.new_stage(40).keep(fix, contact).sothat(lift_left, touch, beright)
    ).execute()

    return scene