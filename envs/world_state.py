import numpy as np
import copy
from typing import List
from dataclasses import dataclass
from functools import lru_cache
from tools.config import merge_inputs

from .end_effector import FACTORY
from .soft_utils import empty_soft_body_state, sample_objects
from tools.config import CN, merge_inputs

from sklearn.cluster import DBSCAN
from pytorch3d.transforms import quaternion_apply
from tools.utils import totensor
#import open3d as o3d
from typing import Optional
from numpy.typing import NDArray



@dataclass
class WorldState:
    #particles: List[np.ndarray]
    X: np.ndarray
    V: np.ndarray
    F: np.ndarray
    C: np.ndarray
    qpos: NDArray[np.float32]
    tool_name: str
    rigid_bodies: Optional[NDArray[np.float32]] = None
    ids: NDArray[np.intp] = None
    color: Optional[np.ndarray] = None
    softness: float = 666.
    E_nu_yield: Optional[NDArray[np.float32]] = None

    tool_cfg: Optional[CN] = None


    @property
    def n(self):
        return len(self.X)

    def serialize(self):
        return [self.X, self.V, self.F, self.C, self.qpos, self.tool_name, self.rigid_bodies, self.ids, self.color, self.E_nu_yield]

    @classmethod
    def KEYS(cls):
        return list(cls.__annotations__.keys())

    def allclose(self, other: "WorldState", return_keys=False, *args, **kwargs):
        for k, x, y in zip(self.KEYS(), self.serialize(), other.serialize()):
            if return_keys:
                print(k)
            if k == 'rigid_bodies' or k == 'E_nu_yield':
                # 'rigid_bodies' is only used for the sanity check.
                continue
            if k == 'tool_name':
                incorrect = (x != y)
            else:
                if k == 'rigid_bodies':
                    print('x', x, 'y', y)
                incorrect = not np.allclose(x, y, *args, **kwargs)
            if incorrect:
                if return_keys:
                    print(x, y)
                    return k
                return False
        if return_keys:
            return None
        return True

    @classmethod
    def get_empty_state(cls, n=None, tool_name='Gripper', ids=None, **kwargs):
        #self.tool_cur = self.tools[tool_name]
        # WorldState.get_empty_state(pcd: ndarray(N x 3))
        tool_cls = FACTORY[tool_name]

        E_nu_yield = kwargs.pop('E_nu_yield') if 'E_nu_yield' in kwargs else None

        soft_body_state = empty_soft_body_state(n, **kwargs)
        n = len(soft_body_state['particles'][0])
        tool_state = tool_cls.empty_state()

        if ids is None:
            ids = np.zeros(n, dtype=np.int32)

        state = WorldState(
            *soft_body_state['particles'], tool_state, tool_name,
            None, ids,
            soft_body_state['color'], soft_body_state['softness'],
            E_nu_yield, None,
        )
        return state

    def get_empty_state_qpos(self):
        self.rigid_bodies = None
        
        if self.tool_name == 'Fingers':
            self.tool_name = 'Pusher'
            self.tool_cfg = None

        return FACTORY[self.tool_name].empty_state()

    @classmethod
    def sample_shapes(cls, obj_cfg=None, **kwargs):
        obj_cfg = merge_inputs(obj_cfg, **kwargs)
        out, _ = sample_objects(obj_cfg)

        return cls.get_empty_state(init=out['pos'], ids=out['ids'], color=out['colors'], E_nu_yield=out['properties'])

    def copy(self):
        return copy.copy(self)

    def switch_tools(self, tool_name, qpos=None, **kwargs):
        state = self.copy()

        no_switch_tool = tool_name == state.tool_name

        state.tool_name = tool_name
        state.rigid_bodies = None
        state.qpos = qpos if qpos is not None else FACTORY[tool_name].empty_state()
        
        state.tool_cfg = merge_inputs(self.tool_cfg if no_switch_tool else None, **kwargs)
        return state

    def add_state(self, state):
        #print(self.E_nu_yield.shape, state.E_nu_yield.shape)
        # return [self.X, self.V, self.F, self.C, self.qpos, self.tool_name, self.rigid_bodies, self.ids, self.color, self.E_nu_yield]
        E_nu_yield = state.E_nu_yield
        if E_nu_yield is None:
            E_nu_yield = np.zeros((len(state.X), 3)) - 1.
        return WorldState(
            np.concatenate([self.X, state.X], axis=0),
            np.concatenate([self.V, state.V], axis=0),
            np.concatenate([self.F, state.F], axis=0),
            np.concatenate([self.C, state.C], axis=0),
            #np.concatenate([self.qpos, state.qpos], axis=0),
            self.qpos,
            self.tool_name,
            self.rigid_bodies,
            np.concatenate([self.ids, state.ids], axis=0),
            np.concatenate([self.color, state.color], axis=0),
            self.softness,
            np.concatenate([self.E_nu_yield, E_nu_yield], axis=0),
            self.tool_cfg,
        )
        
        
        
    def add_part(self, id, N, en, com, p, q):
        self.X = np.concatenate((self.X, self.X[:N]), axis=0)
        self.V = np.concatenate((self.V, self.V[:N]), axis=0)
        self.F = np.concatenate((self.F, self.F[:N]), axis=0)
        self.C = np.concatenate((self.C, self.C[:N]), axis=0)
        self.ids = np.concatenate((self.ids, self.ids[:N]), axis=0)
        self.color = np.concatenate((self.color, self.color[:N]), axis=0)
        self.E_nu_yield = np.concatenate((self.E_nu_yield, self.E_nu_yield[:N]), axis=0)
        
        com, p, q = np.asarray(com.cpu()), np.asarray(p), np.asarray(q)
        # print(com, p, q)
        tp = self.X[:,1].copy()
        self.X[:,1] = self.X[:,2].copy()
        self.X[:,2] = tp.copy()

        self.X[:N] = np.asarray(quaternion_apply(totensor(q, 'cuda:0'), totensor(self.X[:N] - com, 'cuda:0')).cpu()) + com + p

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.X[en:])
        dpcd = pcd.voxel_down_sample(voxel_size=0.01) #0.0075)
        nx = np.asarray(dpcd.points)
        nn = nx.shape[0]

        self.X = np.concatenate((self.X[:en], nx), axis=0)
        self.V = np.concatenate((self.V[:en], self.V[en:en+nn]), axis=0)
        self.F = np.concatenate((self.F[:en], self.F[en:en+nn]), axis=0)
        self.C = np.concatenate((self.C[:en], self.C[en:en+nn]), axis=0)
        self.ids = np.concatenate((self.ids[:en], self.ids[en:en+nn]), axis=0)
        self.color = np.concatenate((self.color[:en], self.color[en:en+nn]), axis=0)
        self.E_nu_yield = np.concatenate((self.E_nu_yield[:en], self.E_nu_yield[en:en+nn]), axis=0)

        tp = self.X[:,1].copy()
        self.X[:,1] = self.X[:,2].copy()
        self.X[:,2] = tp.copy()



    def check_soft_bodies(self, cen, rad):
        inn = ((self.X - cen) ** 2).sum(axis=1)         
        inn = inn > rad * rad
        self.X = self.X[inn]
        self.V = self.V[inn]
        self.F = self.F[inn]
        self.C = self.C[inn]
        self.ids = self.ids[inn]
        self.color = self.color[inn]
        self.E_nu_yield = self.E_nu_yield[inn]
        if self.X.size == 0:
            return

        # print(self.X.shape, self.V.shape, self.F.shape, self.C.shape, self.ids.shape, self.color.shape, self.E_nu_yield.shape)
        dbscan = DBSCAN(eps=0.1, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
        dbscan.fit(self.X)
        lbl = np.int32(dbscan.labels_) + 1

        nids = self.ids.copy()
        curid = 0
        for i in np.unique(self.ids):
            ids = np.where(self.ids == i)[0]
            #print(self.X.shape, ids.shape, lbl.shape)
            if np.all(lbl[ids] == lbl[ids[0]]):
                nids[ids] = curid
                curid += 1
            else:
                for j in np.unique(lbl[ids]):
                    idss = np.where(lbl[ids] == j)[0]
                    nids[idss] = curid
                    curid += 1
                

        self.ids = nids.copy() #np.int32(dbscan.labels_) + 1

        # for i in np.unique(self.ids):
        #     ids = self.ids.where(i)
        #     ids = np.random.choice(ids, 1500)
        #     if i == 0:
        #         nX = self.X[ids]
        #         nV = self.V[ids]
        #         nF = self.F[ids]
        #         nC = self.C[ids]
        #         nids = self.ids[ids]
        #         ncolor = self.color[ids]
        #         nE_nu_yield = self.E_nu_yield[ids]
        #     else:
        #         nX = np.concatenate((nX, self.X[ids]), axis=0)
        #         nV = np.concatenate((nV, self.V[ids]), axis=0)
        #         nF = np.concatenate((nF, self.F[ids]), axis=0)
        #         nC = np.concatenate((nC, self.C[ids]), axis=0)
        #         nids = np.concatenate((nids, self.ids[ids]), axis=0)
        #         ncolor = np.concatenate((ncolor, self.color[ids]), axis=0)
        #         nE_nu_yield = np.concatenate((nE_nu_yield, self.E_nu_yield[ids]), axis=0)

        # self.X = nX
        # self.V = nV
        # self.F = nF
        # self.C = nC    
        # self.ids = nids 
        # self.color = ncolor 
        # self.E_nu_yield = nE_nu_yield

        # print('origin labels', np.unique(self.ids))
        # uni, cnt = np.unique(np.int32(dbscan.labels_), return_counts=True)
        # print('dbscan labels', np.asarray((uni, cnt)).T)
        # # print(dbscan.components_.shape)
        # from llm.envs.soft_utils import rgb2int
        # for i, col in enumerate(self.color):
        #     lb = dbscan.labels_[i] + 1
        #     #print(self.color[i])
        #     if lb % 5 == 0:
        #         self.color[i] = rgb2int(255,0,0)
        #     elif lb % 5 == 1:
        #         self.color[i] = rgb2int(0,255,0)
        #     elif lb % 5 == 2:
        #         self.color[i] = rgb2int(0,0,255)
        #     elif lb % 5 == 3:
        #         self.color[i] = rgb2int(255,255,255)
        #     elif lb % 5 == 4:
                # self.color[i] = rgb2int(255,0,0)
        # exit(0)
        # print('labels', dbscan.labels_)

    def reid(self):
        dbscan = DBSCAN(eps=0.07, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
        dbscan.fit(self.X)
        lbl = np.int32(dbscan.labels_)
        print(lbl)
        nids = self.ids.copy()
        curid = 0
        for i in np.unique(self.ids):
            ids = np.where(self.ids == i)[0]
            # print(self.X.shape, ids.shape, lbl.shape)
            if np.all(lbl[ids] == lbl[ids[0]]):
                nids[ids] = curid
                curid += 1
            else:
                for j in np.unique(lbl[ids]):
                    idss = np.where(lbl[ids] == j)[0]
                    nids[idss] = curid
                    curid += 1
        self.ids = nids.copy() 

    def merge(self):
        dbscan = DBSCAN(eps=0.07, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
        dbscan.fit(self.X)
        lbl = np.int32(dbscan.labels_)
        self.ids = lbl.copy()
