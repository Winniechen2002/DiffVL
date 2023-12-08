import torch
import numpy as np
from tools.utils import totensor
from .mydsl import dsl, color_type, myint, List, lcurry, compose, lmap, rcurry, apply, car, cdr
from ..pl.types import Type, DataType, Arrow
from .libpcd import ArrayType, PointCloudType, myfloat
from .libpcd import Array2D

class SoftBody:
    # soft body is
    def __init__(self, scene, indices):
        from .scene import Scene
        self.scene: Scene = scene
        self.trajs = self.scene.obs

        self.indices = indices

        self.object_id = self.scene.initial_state.ids[indices[0]]
        self._color = self.scene.initial_state.color[np.array(indices.cpu())]
        self._pcd = None
        self.init_T = self.scene.cur_step

    @classmethod
    def new(cls, scene, object_id):
        indices = torch.tensor(
            np.where(object_id == scene.initial_state.ids)[0], device='cuda:0', dtype=torch.long)
        return SoftBody(scene, indices)


    @dsl.as_attr
    def color(self) -> color_type:
        return self._color

    def rgba(self):
        #return self._color
        import numpy as np
        #c = COLOR_MAP[self._color]
        #return np.array([(c//256//256) % 256, c//256 % 256, c % 256, 0])
        c = self._color[0]
        return np.array([(c//256//256) % 256, c//256 % 256, c % 256, 0])

    def get_color(self):
        #return self._color
        import numpy as np
        #c = COLOR_MAP[self._color]
        #return np.array([(c//256//256) % 256, c//256 % 256, c % 256, 0])
        c = self._color
        ret = np.zeros((c.shape[0],4))
        ret[:,0] = (c//256//256) % 256
        ret[:,1] = c//256 % 256
        ret[:,2] = c% 256
        ret[:,3] = 0.001
        return ret

    @dsl.as_attr
    def N(self) -> myint:
        return len(self.indices)

    @dsl.as_attr
    def pcd(self) -> PointCloudType:
        # TODO: cache ..
        return self.scene.get_obs()['pos'][self.indices]

    @dsl.as_attr
    def t(self) -> myint:
        return self.scene.cur_step

    # --------------------------------------------------------------------------------------------------
    # 先用Primitive来凑合
    @dsl.as_attr
    def left(self):
        pcd = self.pcd()
        com = pcd.mean(axis=0)
        ind = self.indices[pcd[:, 0] < com[0]]
        # raise NotImplementedError
        return SoftBody(self.scene, ind)


    @dsl.as_attr
    def right(self):
        pcd = self.pcd()
        com = pcd.mean(axis=0)
        ind = self.indices[pcd[:, 0] >= com[0]]
        return SoftBody(self.scene, ind)

    @dsl.as_attr
    def back(self):
        pcd = self.pcd()
        com = pcd.mean(axis=0)
        ind = self.indices[pcd[:, 2] > com[2]]
        return SoftBody(self.scene, ind)

    @dsl.as_attr
    def front(self):
        pcd = self.pcd()
        com = pcd.mean(axis=0)
        ind = self.indices[pcd[:, 2] < com[2]]
        return SoftBody(self.scene, ind)

    @dsl.as_attr
    def up(self):
        pcd = self.pcd()
        com = pcd.mean(axis=0)
        ind = self.indices[pcd[:, 1] > com[1]]
        return SoftBody(self.scene, ind)

    @dsl.as_attr
    def rightmost(self, eps: myfloat):
        pcd = self.pcd()
        bound = pcd.max(axis=0)[0][0]
        ind = self.indices[pcd[:, 0] >= bound - eps]
        return SoftBody(self.scene, ind)

    @dsl.as_attr
    def leftmost(self, eps: myfloat):
        pcd = self.pcd()
        bound = pcd.min(axis=0)[0][0]
        ind = self.indices[pcd[:, 0] <= bound + eps]
        return SoftBody(self.scene, ind)

    @dsl.as_attr
    def upmost(self, eps: myfloat):
        pcd = self.pcd()
        bound = pcd.max(axis=0)[0][1]
        ind = self.indices[pcd[:, 1] >= bound - eps]
        return SoftBody(self.scene, ind)

    @dsl.as_attr
    def surface(self):
        """
        Consider a naive surface extraction algorithm .. 
        for each particle find its nearest one ..
        """
        with torch.no_grad():
            from pytorch3d.ops import ball_query
            R = 0.12
            delta = totensor(np.array([
                [-1., 0., 0.],
                [1., 0., 0.],
                [0., -1., 0.],
                [0., 1., 0.],
                [0., 0., -1.],
                [0., 0., 1.],
            ]), device='cuda:0')
            delta = delta / torch.linalg.norm(delta, axis=-1)[:, None]

            pcd = self.pcd()

            queries = pcd[:, None] + delta[None, :] * (R + 0.01)
            #raise NotImplementedError
            _, idx, _ = ball_query(queries.reshape(
                1, -1, 3), pcd[None, :], K=1, radius=R+0.009, return_nn=True)
            ind = self.indices[(idx < 0).reshape(len(pcd), -1).any(axis=1)]

        return SoftBody(self.scene, ind)


    @dsl.as_attr
    def near_pairs(self, eps: myfloat) -> Array2D:
        with torch.no_grad():
            from pytorch3d.ops import ball_query
            R = 0.02 #12
            delta = totensor(np.array([
                [-1., 0., 0.],
                [1., 0., 0.],
                [0., -1., 0.],
                [0., 1., 0.],
                [0., 0., -1.],
                [0., 0., 1.],
            ]), device='cuda:0')
            delta = delta / torch.linalg.norm(delta, axis=-1)[:, None]

            pcd = self.pcd()

            queries = pcd[:, None] # + delta[None, :] * (R + 0.01)
            #raise NotImplementedError
            _, idx, _ = ball_query(queries.reshape(
                1, -1, 3), pcd[None, :], K=10, radius=R+0.009, return_nn=True)
            ind = idx.squeeze(0).reshape(-1) # print(ind)
            ind = torch.vstack([totensor([_ for _ in range(len(pcd)) for i in range(10)], device='cuda:0'), ind])
            # ind = ind.transpose(0, 1)
            # print(ind.shape)
        return ind

    @dsl.as_attr
    def no_break(self, ind: Array2D) -> myfloat:
        pcd = self.pcd()
        # pcd.register_hook(lambda grad: print('pcd', grad.mean()))
        # ind.to(torch.long)
        mx, _ = torch.linalg.norm(pcd[ind[0].long()] - pcd[ind[1].long()], axis=-1).max(axis=-1)
        # mx.register_hook(lambda x: print('mx', x))
        return mx



soft_body_type = dsl.build_data_type(SoftBody)