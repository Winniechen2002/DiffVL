import torch
import numpy as np
from llm.pl.types import Type, Arrow, VariableArgs
from .mydsl import dsl, List, myint, mybool


ACCURATE = 0.001
ROUGHLY = 0.05


tA = Type('\'A')
tB = Type('\'B')
tC = Type('\'C')
attr = Arrow(tA, tB)

class BaseInt(Type):
    def __init__(self, a):
        self.a = a

    def __str__(self):
        return str(self.a)

    def instance(self, value):
        return isinstance(value, int) and value == self.a 

class SizeType(Type):
    def __init__(self, *size):
        self.dot = None # (int, ..., int) something like this ..
        self.size = []
        for idx, i in enumerate(size):
            if isinstance(i, int):
                i = BaseInt(i)
            elif i.match_many():
                assert self.dot is None, NotImplementedError("only one ellipsis is allowed.")
                self.dot = idx
            else:
                assert isinstance(i, BaseInt) or hasattr(i, '_type_name'), f"{i} of {type(i)}"
            self.size.append(i)

    def __str__(self):
        return '(' + ', '.join(map(str, self.size)) + ')'

    def instance(self, size):
        if len(size) != len(self.size) or (self.dot is not None and len(size) < len(self.size)):
            return False

        def match(A, B):
            if len(A) != len(B):
                return False
            for a, b in zip(A, B):
                if not b.instance(a):
                    return False
            return True
        if self.dot is None:
            return match(size, self.size)
        else:
            l = self.dot
            r = len(self.size) - l
            if l > 0 and not match(size[:l], self.size[:l]):
                return False
            if r > 0 and not match(size[-r:], self.size[-r:]):
                return False
            return match(size[l:-r], self.size[l])

    @property
    def children(self):
        return tuple(self.size)
    

class ArrayType(Type):
    BASETYPE = 'Array'
    def __init__(self, *size):
        size = SizeType(*size)
        assert isinstance(size, SizeType) or hasattr(size, '_type_name') # size could be either an size type or a type variable ..
        self.size = size
        self.data_cls = torch.Tensor
        self.type_name = 'Tensor' + str(size)

    def instance(self, value):
        if not (isinstance(value, self.data_cls)):
            return False
        return self.size.instance(value.shape)

    @property
    def children(self):
        return self.size.children

    def __str__(self):
        return self.type_name


TN = Type('\'N')
TM = Type('\'M')
TN1 = Type('\'N1')
Array2D_A = ArrayType(TN, TM)
Array2D_B = ArrayType(TN1, TM)

PointCloudType = ArrayType(TN, 3)
Array2D = ArrayType(TN, TM)

PositionType = ArrayType(3)
Array1D = ArrayType(TN)

ellipsis = VariableArgs('\'*') # ellipsis can match any size ..
ellipsisB = VariableArgs('\'**')
Array = ArrayType(ellipsis)


myfloat = ArrayType()
dsl.register_constant_type(float, myfloat, lambda token: torch.tensor(token, dtype=torch.float32, device='cuda:0'))


__chamfer = None
@dsl.as_primitive
def chamfer(x: Array2D_A, y: Array2D_B) -> myfloat:
    global __chamfer
    from chamferdist import ChamferDistance
    __chamfer = __chamfer or ChamferDistance()
    return __chamfer(x[None,:].clone(), y[None,:].clone())/len(x)

_emd_fn = None
@dsl.as_primitive
def emd(a: Array2D_A, b: Array2D_B) -> myfloat:
    p=1; blur=0.01; #, *args, **kwargs
    global _emd_fn
    from geomloss import SamplesLoss
    _emd_fn = _emd_fn or SamplesLoss(loss='sinkhorn', p=2., blur=0.01)
    _emd_fn.p = p
    _emd_fn.blur = blur
    return _emd_fn(a.clone(), b.clone())


    
@dsl.as_primitive
def contact_distance(a: Array2D_A, b: Array2D_B, T: myfloat) -> myfloat:
    dist = torch.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).reshape(-1)
    softmin = torch.nn.functional.softmin(dist * T, -1)
    return  (dist * softmin).sum()


@dsl.as_primitive
def nearest_dist(a: Array2D_A, b: Array2D_B) -> myfloat:
    from pytorch3d.ops import knn_points
    nearest =  knn_points(a[None,:], b[None,:], K=1)[0]
    min_dist = nearest[0, :, 0].min(axis=0).values ** 0.5
    return min_dist


@dsl.as_primitive
def transport(a: ArrayType(ellipsis, 3), b: ArrayType(3)) -> ArrayType(ellipsis, 3):
    out = a + b
    assert out.shape == a.shape
    return out


@dsl.as_primitive
def pcd_dist(a: Array2D, b: Array2D) -> Array1D:
    assert a.shape == b.shape and a.shape[-1] == 3
    return (((a - b) ** 2).sum(axis=-1) ** 0.5)

@dsl.as_primitive
def max_array(a: Array) -> myfloat:
    return a.max()

@dsl.as_primitive
def min_array(a: Array) -> myfloat:
    return a.min()

@dsl.as_primitive
def mean_array(a: Array) -> myfloat:
    return a.mean()
        

@dsl.as_primitive
def save_first_time(a: Array) -> Array:
    # save to pcd ..
    raise NotImplementedError

@dsl.as_primitive
def __Array__com(tool: ArrayType(ellipsis, 3)) -> PositionType:
    while len(tool.shape) > 1:
        tool = tool.mean(axis=0, keepdims=False)
    return tool

@dsl.as_primitive
def __Array__max(tool: ArrayType(ellipsis, 3)) -> PositionType:
    while len(tool.shape) > 1:
        tool = tool.max(axis=0, keepdims=False)[0]
    return tool

@dsl.as_primitive
def __Array__min(tool: ArrayType(ellipsis, 3)) -> PositionType:
    while len(tool.shape) > 1:
        tool = tool.min(axis=0, keepdims=False)[0]
    return tool


@dsl.as_primitive
def __Array__get(a: ArrayType(ellipsis, tA), b: myint) -> ArrayType(ellipsis):
    return a[..., b]

@dsl.as_primitive
def __Array__x(a: ArrayType(ellipsis, 3)) -> ArrayType(ellipsis):
    return a[..., 0]

@dsl.as_primitive
def __Array__y(a: ArrayType(ellipsis, 3)) -> ArrayType(ellipsis):
    return a[..., 1]

@dsl.as_primitive
def __Array__z(a: ArrayType(ellipsis, 3)) -> ArrayType(ellipsis):
    return a[..., 2]

@dsl.as_primitive
def __Array__norm(a: Array) -> myfloat:
    return torch.linalg.norm(a)

@dsl.as_primitive
def relative_pose(a: PointCloudType, b: PointCloudType) -> ArrayType(3, 4):
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    P = a[None, :]
    Q = b[None, :]

    P1=P-P.mean(dim=1,keepdim=True)
    Q1=Q-Q.mean(dim=1, keepdim=True)
    C = torch.bmm(P1.transpose(1,2), Q1)## B*3*3

    U, S, V = torch.svd(C)
    U = U.transpose(1, 2)
    d = (torch.det(V) * torch.det(U)) < 0.0
    d = - d.float().detach() * 2 + 1 # d == 1 -> -1; d == 0 -> 1
    V = torch.cat((V[:,:-1],  d[:, None, None] * V[:,-1:]), dim=1)
    R = torch.bmm(V, U)


    # T=Q-torch.bmm(R,P.transpose(1,2)).transpose(2,1)
    cors= torch.bmm(R, P.transpose(1, 2)).transpose(2, 1)
    T = (Q - cors).mean(dim=-2)
    return torch.cat((R, T[:,:,None]), dim=2)[0] # return a 3x4 matrix

@dsl.as_primitive
def pose2rpy(relative_pose: ArrayType(3, 4), axis: myint) -> myfloat:
    from pytorch3d.transforms import matrix_to_euler_angles
    axis = torch.tensor(np.array(axis), device=relative_pose.device)
    return matrix_to_euler_angles(relative_pose[:3,:3], 'XYZ')[axis]


@dsl.as_primitive
def pose_dist(relative_pose: Array, rot_axis_angle: Array) -> myfloat:
    from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
    rot_matrix = axis_angle_to_matrix(rot_axis_angle)
    return torch.arccos((torch.trace(torch.linalg.inv(relative_pose[:3,:3]) @ rot_matrix) - 1)/2)

#@dsl.as_primitive
#def as_tensor(a: List(myfloat)) -> Array1D:
#    return torch.stack(a)

@dsl.as_primitive
def as_pos(x: myfloat, y: myfloat, z: myfloat) -> PositionType:
    return torch.stack([x, y, z])


@dsl.as_primitive
def concat_pcd(x: ArrayType(ellipsis), y: ArrayType(ellipsis)) -> ArrayType(ellipsis):
    return torch.concat([x, y])
    
@dsl.as_primitive
def dist2grid(a: PositionType, x: myint, y: myint, z: myint) -> myfloat:
    t = torch.tensor([x, y, z], device=a.device).float() / 32.
    return torch.linalg.norm(a - t)