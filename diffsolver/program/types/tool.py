import trimesh
from .softbody import SoftBody
from tools.utils import totensor
from .scene import SceneSpec
from pytorch3d.transforms import quaternion_apply

from trimesh.creation import capsule

def capsule2pcd(size):
    r=size[0]; h=size[1]
    mesh = capsule(height=h, radius = r)
    v = mesh
    return trimesh.Trimesh(v.vertices[:, [0, 2, 1]], v.faces)


class Tool:
    def __init__(self, scene: SceneSpec):
        self.scene = scene
        self.trajs = self.scene.obs
        cur = scene.env.tool_cur
        assert cur is not None
        self.tool_cur = cur
        self.tool_type = self.tool_cur.name
        self.pcds = {}

    def make_pcd(self):
        if 'capsule' not in self.pcds:
            assert self.tool_cur._cfg is not None
            pcd = capsule2pcd(self.tool_cur._cfg.size)
            self.pcds['capsule'] = totensor(pcd.vertices, 'cuda:0')[::5]
        return self.pcds['capsule']

    def pcd(self):
        state = self.scene.get_obs()['tool'][0]
        pcd = quaternion_apply(state[3:7], self.make_pcd()) + state[:3]
        return pcd

    def rpy(self):
        return self.scene.get_obs()['qpos'][3:6]

    def qpos(self):
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