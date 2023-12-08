from llm.tiny import *

list_float = List(myfloat)
string = dsl.str

@dsl.as_func
def still_fn(x: soft_body_type):
    x_ = x.pcd()
    return lambda: max_array(pcd_dist(x.pcd(), x_)) < 0.02

@dsl.as_func
def contact_obj_fn(tool: tool_type, obj: soft_body_type):
    return lambda: tool.dist2body(obj) < 0.01

@dsl.as_func
def shape_l2_reach_fn(obj: soft_body_type, pcd2: PointCloudType):
    return lambda: mean_array(pcd_dist(obj.pcd(), pcd2)) < 0.005

#TODO: shape distance jk

@dsl.as_func
def make_stable(scene: scene_type, T: myint):
    obj_all = scene.obj(-1)
    return scene.new_stage(T).sothat(still_fn(obj_all)).execute()

@dsl.as_func
def pickup(scene: scene_type,
           obj: soft_body_type,
           dir: list_float,
           T: myint,
           target_height: myfloat):
    scene.grasp(obj, dir)
    obj_pcd = obj.pcd()
    dy = target_height - min_array(obj_pcd.y())
    target = transport(obj_pcd, as_pos(0., dy, 0.)) # lift up for 0.2
    contact = contact_obj_fn(scene.tool(), obj)
    return scene.new_stage(10).sothat(contact).then(
        scene.new_stage(T).keep(contact).sothat(
            shape_l2_reach_fn(obj, target),
        )
    ).execute()

@dsl.as_func
def place(scene: scene_type, obj: soft_body_type, obj2: soft_body_type, T: myint):
    # com coincidents ..
    obj2_com = obj2.pcd().com()
    top = obj2.upmost(0.02)
    center = as_pos(obj2_com.x(), 0., obj2_com.z())

    match_center = lambda: (
        as_pos(obj.pcd().com().x(), 0., obj.pcd().com().z()) - center
    ).norm() < 0.01

    object_contact = lambda: nearest_dist(top.pcd(), obj.pcd()) < 0.02
    contact = contact_obj_fn(scene.tool(), obj)
    above = lambda: min_array(obj.pcd().y()) > max_array(obj2.pcd().y()) + 0.03

    return scene.new_stage(T).keep(contact).sothat(
        match_center, above
    ).then(
        scene.new_stage(10).sothat(match_center, object_contact)
    ).execute()



@dsl.as_func
def stable_release(scene: scene_type): 
    obj = scene.obj(-1)
    release = lambda: scene.tool().dist2body_min(obj) > 0.05
    return scene.new_stage(20).sothat(still_fn(obj), release).execute()


@dsl.as_func
def pick_and_place(scene: scene_type, obj1: soft_body_type, obj2: soft_body_type, delta: myfloat, T:myint):
    target_height = max_array(obj2.pcd().y()) + delta
    pickup(scene, obj1, [0., 0., 0.], T, target_height)
    place(scene, obj1, obj2, T)
    return stable_release(scene)


@dsl.as_primitive
def load_shape(a: string) -> PointCloudType:
    if a.endswith('plt') or a.endswith('pcd'):
        import open3d as o3d
        from tools.utils import totensor
        return totensor(np.asarray(o3d.io.read_point_cloud(a).points), device='cuda:0')
    else:
        raise NotImplementedError


@dsl.as_primitive
def loop(a: tA, T: myint) -> none:
    for i in range(T):
        a()

@dsl.as_primitive
def ifelse(a: mybool, b: tA, c: tA) -> none:
    if a:
        b()
    else:
        c()


@dsl.as_func
def deform_into(scene: scene_type, goal: PointCloudType):
    return 0

@dsl.as_func
def divide(obj: soft_body_type, cut_plane: PositionType, ratio: myfloat):
    return 0