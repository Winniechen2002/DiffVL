from llm.genetic import *

def hang(scene: Scene):
    """
    scene: rope_hang.yml
    env_cfg:
        sim_cfg:
            ground_friction: 0.1
    """
    rod = scene.obj(0)
    rope = scene.obj(1)
    rope_right_end = rope.rightmost(0.05)
    tool = scene.tool()

    contact = contact_obj_fn(scene.tool(), rope_right_end)
    rod_initial = rod.pcd()
    fix = lambda: chamfer(rod.pcd(), rod_initial) < 0.0002

    rope_middle = rope.right().leftmost(0.05)

    across = lambda: max_array(rope_right_end.pcd().com().x()) > rod.pcd().com().x() + 0.05

    lift = lambda: min_array(rope_middle.pcd().y()) > max_array(rod.pcd().y())
    touch = lambda: (rope_middle.pcd().com() - rod.upmost(0.05).pcd().com()).norm() < 0.05


    scene.grasp(rope_right_end, [0., 1., 0.])
    scene.new_stage(10).sothat(contact).then(
        scene.new_stage(30).keep(contact).sothat(lift)
    ).execute()
    scene.new_stage(40).keep(contact, fix).sothat(
        across, touch
    ).execute()

    return scene