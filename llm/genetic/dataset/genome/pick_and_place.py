from llm.genetic import *


def push_right(scene: Scene):
    """
    scene:  wrap.yml
    """
    box = scene.obj(0)
    mat = scene.obj(1)
    tool = scene.tool()

    _ = pick_and_place(scene, box, mat, 0.1, 25)
    
    
    box_pcd = box.pcd()
    mat_part = mat.left().left().back().back()

    grasp_mat_corner = lambda: tool.dist2body(mat) < 0.005
    box_stable = lambda: max_array(pcd_dist(box.pcd(), box_pcd)) < 0.05
    lift_mat_corner = lambda: max_array(mat_part.pcd().y()) > 0.25
    gripper_is_verticle = lambda: (tool.rpy() - as_pos(0., 0.5, 0.)).norm() < 0.05
    cover_the_box = lambda: chamfer(mat_part.pcd(), box.up().pcd()) < 0.002

    scene.grasp(mat_part, [0., 0.5, 1.])
    scene.new_stage(10).sothat(
        grasp_mat_corner
    ).then(
        scene.new_stage(40).keep(box_stable).sothat(lift_mat_corner, gripper_is_verticle)
    ).then(
        scene.new_stage(30).keep(box_stable).sothat(cover_the_box)
    ).then(
        scene.new_stage(20).keep(box_stable).sothat(cover_the_box)
    ).execute()
    return scene