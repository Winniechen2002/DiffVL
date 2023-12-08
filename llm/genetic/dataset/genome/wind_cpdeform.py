from llm.genetic import *

def taco_worm(scene: Scene):
    """
    scene: wind_gripper.yml
    env_cfg:
        sim_cfg:
            ground_friction: 1.
    """
    scene2 = subgoal('dataset/drawing/windrope/3.th')
    # scene2 = subgoal(__SUBGOAL__)

    obj1_left_target = scene2.obj(1).left().pcd()
    obj1_right_target = scene2.obj(1).right().pcd()
    obj1_target = scene2.obj(1).pcd()
    obj0_target = scene2.obj(0).pcd()

    tool = scene.tool()
    obj1 = scene.obj(1)
    obj0 = scene.obj(0)

    scene.cpdeform(obj1.left(), obj1_left_target, [1., 0., 0., 0., 0., 0.])

    contact =lambda: tool.dist2body(obj1) < 0.005
    fix = lambda: chamfer(obj0.pcd(), obj0_target) < 0.001

    scene.new_stage(10).sothat(contact).then(scene.new_stage(40).keep(
        contact, fix).sothat(
        lambda: emd(obj1.left().pcd(), obj1_left_target) < 0.005,
    )).execute()

    stable_release(scene)
    #scene.cpdeform(obj1, obj1_target, [1., 0., 0., 1., 1., 1.])
    scene.grasp(obj1.rightmost(0.05), [0., 1., 0.])
    # render_rgb(obj1.right(), 'xx.png')
    obj1_right_most = obj1.rightmost(0.02)

    contact_right =lambda: tool.dist2body(obj1_right_most) < 0.005
    scene.new_stage(30).sothat(contact_right).then(scene.new_stage(40).keep(
        contact_right, fix).sothat(
        lambda: emd(obj1.right().pcd(), obj1_right_target) < 0.005,
    )).execute()

    return scene