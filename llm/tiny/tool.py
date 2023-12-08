from .mydsl import dsl, color_type, myint, List, lcurry, compose, lmap, rcurry, apply, car, cdr, none, EnumType
from .libpcd import myfloat
from .softbody import SoftBody, soft_body_type
from tools.utils import totensor
from .libpcd import PositionType, PointCloudType, ArrayType, Array1D

from pytorch3d.transforms import quaternion_apply

class Tool:
    def __init__(self, scene):
        from .scene import Scene
        self.scene = scene
        self.trajs = self.scene.obs
        self.tool_type = scene.env.tool_cur.name
        self.pcds = {}

    def cur_tool(self):
        return self.scene.env.tool_cur

    def make_pcd(self):
        if 'capsule' not in self.pcds:
            from .utils import capsule2pcd
            pcd = capsule2pcd(self.cur_tool()._cfg.size)
            self.pcds['capsule'] = totensor(pcd.vertices, 'cuda:0')[::5]
        return self.pcds['capsule']

    @dsl.as_attr
    def pcd(self) -> PointCloudType:
        state = self.scene.get_obs()['tool'][0]
        pcd = quaternion_apply(state[3:7], self.make_pcd()) + state[:3]
        return pcd

    @dsl.as_attr
    def rpy(self)  -> ArrayType(3):
        #return self.scene.get_obs()['tool'][0][3:7]
        return self.scene.get_obs()['qpos'][3:6]


    @dsl.as_attr
    def qpos(self) -> Array1D:
        #return self.scene.get_obs()['tool'][0][3:7]
        return self.scene.get_obs()['qpos']

    # @dsl.as_attr
    # def left(self) -> Array:
    #     state = self.obs['tool'][0]
    #     assert state.shape == (7,)
    #     p = self.make_pcd()
    #     p = p[p[:, 0] > 0]
    #     p = p[p[:, 1] < 0]
    #     return quaternion_apply(state[3:7], p) + state[:3]
        
    # @dsl.as_attr
    # def right(self) -> Array:
    #     assert self.tool_type == 'Gripper'
    #     state = self.obs['tool'][1]
    #     p = self.make_pcd()
    #     p = p[p[:, 0] < 0]
    #     p = p[p[:, 1] < 0]
    #     return quaternion_apply(state[3:7], p) + state[:3]


tool_type = dsl.build_data_type(Tool)
tool_name_type = EnumType(['Gripper', 'DoublePushers', 'Pusher', 'Knife', 'Rolling_Pin'], "ToolNameType")

@dsl.as_primitive
def __Tool__dist2body(tool: tool_type, soft_body: soft_body_type) -> myfloat:
    return tool.scene.get_obs()['dist'][soft_body.indices].min(dim=0)[0].sum(dim=0)

@dsl.as_primitive
def __Tool__dist2body_min(tool: tool_type, soft_body: soft_body_type) -> myfloat:
    return tool.scene.get_obs()['dist'][soft_body.indices].min(dim=0)[0].min(dim=0)[0]
