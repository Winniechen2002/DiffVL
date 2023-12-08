from typing import Any, Tuple

import numpy as np

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

import itertools

import numpy as np

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def euler2mat(euler):
    """ Convert Euler Angles to Rotation Matrix.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def mat2quat(mat):
    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))


def subtract_euler(e1, e2):
    assert e1.shape == e2.shape
    assert e1.shape[-1] == 3
    q1 = euler2quat(e1)
    q2 = euler2quat(e2)
    q_diff = quat_mul(q1, quat_conjugate(q2))
    return quat2euler(q_diff)


def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_conjugate(q):
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q


def quat_mul(q0, q1):
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


def quat_rot_vec(q, v0):
    q_v0 = np.array([0, v0[0], v0[1], v0[2]])
    q_v = quat_mul(q, quat_mul(q_v0, quat_conjugate(q)))
    v = q_v[1:]
    return v


def quat_identity():
    return np.array([1, 0, 0, 0])


def quat_difference(q, p):
    return quat_normalize(quat_mul(q, quat_conjugate(p)))


def quat_magnitude(q):
    w = q[..., 0]
    assert np.all(w >= 0)
    return 2 * np.arccos(np.clip(w, -1.0, 1.0))


def quat_normalize(q):
    assert q.shape[-1] == 4
    sign = np.sign(q[..., [0]])
    # Sign takes value of 0 whenever the input is 0, but we actually don't want to do that
    sign[sign == 0] = 1
    return q * sign  # use quat with w >= 0


def quat_average(quats, weights=None):
    """Weighted average of a list of quaternions."""
    n_quats = len(quats)
    weights = np.array([1.0 / n_quats] * n_quats if weights is None else weights)
    assert np.all(weights >= 0.0)
    assert len(weights) == len(quats)
    weights = weights / np.sum(weights)

    # Average of quaternion:
    # https://math.stackexchange.com/questions/1204228
    # /how-would-i-apply-an-exponential-moving-average-to-quaternions
    outer_prods = [w * np.outer(q, q) for w, q in zip(weights, quats)]
    summed_outer_prod = np.sum(outer_prods, axis=0)
    assert summed_outer_prod.shape == (4, 4)

    evals, evecs = np.linalg.eig(summed_outer_prod)
    evals, evecs = np.real(evals), np.real(evecs)
    biggest_i = np.argmax(np.real(evals))
    return quat_normalize(evecs[:, biggest_i])


def quat2axisangle(quat):
    theta = 0
    axis = np.array([0, 0, 1])
    sin_theta = np.linalg.norm(quat[1:])

    if sin_theta > 0.0001:
        theta = 2 * np.arcsin(sin_theta)
        theta *= 1 if quat[0] >= 0 else -1
        axis = quat[1:] / sin_theta

    return axis, theta


def euler2point_euler(euler):
    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 3
    _euler_sin = np.sin(_euler)
    _euler_cos = np.cos(_euler)
    return np.concatenate([_euler_sin, _euler_cos], axis=-1)


def point_euler2euler(euler):
    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 6
    angle = np.arctan(_euler[..., :3] / _euler[..., 3:])
    angle[_euler[..., 3:] < 0] += np.pi
    return angle


def quat2point_quat(quat):
    # Should be in qw, qx, qy, qz
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 4
    angle = np.arccos(_quat[:, [0]]) * 2
    xyz = _quat[:, 1:]
    xyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (xyz / np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
        ]
    return np.concatenate([np.sin(angle), np.cos(angle), xyz], axis=-1)


def point_quat2quat(quat):
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 5
    angle = np.arctan(_quat[:, [0]] / _quat[:, [1]])
    qw = np.cos(angle / 2)

    qxyz = _quat[:, 2:]
    qxyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (qxyz * np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
        ]
    return np.concatenate([qw, qxyz], axis=-1)


def normalize_angles(angles, low=-np.pi, high=np.pi):
    """Puts angles in [low, high] range."""
    angles = angles.copy()
    if angles.size > 0:
        angles = np.mod(angles - low, high - low) + low
        assert low - 1e-6 <= angles.min() and angles.max() <= high + 1e-6
    return angles


def round_to_straight_angles(angles):
    """Returns closest angle modulo 90 degrees """
    angles = np.round(angles / (np.pi / 2)) * (np.pi / 2)
    return normalize_angles(angles)


def round_to_straight_quat(quat):
    angles = quat2euler(quat)
    rounded_angles = round_to_straight_angles(angles)
    return euler2quat(rounded_angles)


def get_parallel_rotations():
    mult90 = [0, np.pi / 2, -np.pi / 2, np.pi]
    parallel_rotations = []
    for euler in itertools.product(mult90, repeat=3):
        canonical = mat2euler(euler2mat(euler))
        canonical = np.round(canonical / (np.pi / 2))
        if canonical[0] == -2:
            canonical[0] = 2
        if canonical[2] == -2:
            canonical[2] = 2
        canonical *= np.pi / 2
        if all([(canonical != rot).any() for rot in parallel_rotations]):
            parallel_rotations += [canonical]
    assert len(parallel_rotations) == 24
    return parallel_rotations


def get_parallel_rotations_180():
    mult180 = [0, np.pi]
    parallel_rotations = []
    for euler in itertools.product(mult180, repeat=3):
        canonical = mat2euler(euler2mat(euler))
        canonical = np.round(canonical / (np.pi / 2))
        if canonical[0] == -2:
            canonical[0] = 2
        if canonical[2] == -2:
            canonical[2] = 2
        canonical *= np.pi / 2
        if all([(canonical != rot).any() for rot in parallel_rotations]):
            parallel_rotations += [canonical]
    assert len(parallel_rotations) == 4
    return parallel_rotations


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape[-1] == 3
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.reshape(angle, axis[..., :1].shape)
    w = np.cos(angle / 2.0)
    v = np.sin(angle / 2.0) * axis
    quat = np.concatenate([w, v], axis=-1)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)

    assert np.array_equal(quat.shape[:-1], axis.shape[:-1])
    return quat


def uniform_quat(random):
    """ Returns a quaternion uniformly at random. Choosing a random axis/angle or even uniformly
    random Euler angles will result in a biased angle rather than a spherically symmetric one.
    See https://en.wikipedia.org/wiki/Rotation_matrix#Uniform_random_rotation_matrices for details.
    """
    w = random.randn(4)
    return quat_normalize(w / np.linalg.norm(w))


def apply_euler_rotations(base_quat, rotation_angles):
    """Apply a sequence of euler angle rotations on to the base quaternion
    """
    new_rot_mat = np.eye(3)
    for rot_angle in rotation_angles:
        new_rot_mat = np.matmul(euler2mat(rot_angle * np.pi / 2.0), new_rot_mat)

    new_rot_mat = np.matmul(quat2mat(base_quat), new_rot_mat)
    new_quat = mat2quat(new_rot_mat)
    return new_quat


def any_orthogonal(vec):
    """ Return any (unit length) vector orthogonal to vec, in a numerically stable way """
    promising_axis = np.eye(3)[np.abs(vec).argmin()]
    non_unit_len = np.cross(vec, promising_axis)

    return non_unit_len / np.linalg.norm(non_unit_len)


def vectors2quat(v_from, v_to):
    """ Define a quaternion rotating along the shortest arc from v_from to v_to """
    q = np.zeros(4)
    dot = np.dot(v_from, v_to)
    v11 = np.dot(v_from, v_from)
    v22 = np.dot(v_to, v_to)
    q[0] = np.sqrt(v11 * v22) + dot
    q[1:4] = np.cross(v_from, v_to)

    if np.linalg.norm(q) < 1e-6:
        # The norm of q is zero if v_from == -v_to, in such case we need to rotate 180 degrees
        # along some not well defined vector orthogonal to both v_from and v_to
        orthogonal = any_orthogonal(v_from)

        q[0] = 0.0  # this is cos(alpha/2) which means rotation 180 deg
        q[1:4] = orthogonal

    return quat_normalize(q / np.linalg.norm(q))


def rot_z_aligned(cube_quat, quat_threshold, include_flip=True):
    """
    Determines if the cube is within quat_threshold of a z-aligned orientation, which means that
    one of the rotatable faces of the **face** cube is on top.
    This means that either the euler angles
    are some pure rotation around z, or they are 180 degree rotation around the x-axis plus some
    rotation around the z.
    """

    cube_angles = quat2euler(cube_quat)
    target_angle = np.eye(3)[-1] * cube_angles

    # if include_flip: True, allow z-axis rotation with top or bottom face as top
    # else, allow z-axis rotation only with top face as top
    if include_flip:
        x_flip = np.asarray([np.pi, 0, 0])
        aligned_angles = [target_angle, target_angle + x_flip]
    else:
        aligned_angles = [target_angle]

    for aligned_angle in aligned_angles:
        aligned_quat = euler2quat(aligned_angle)
        quat_diff = quat_difference(cube_quat, aligned_quat)
        quat_dist = quat_magnitude(quat_diff)
        if quat_dist < quat_threshold:
            return True
    return False


def rot_xyz_aligned(cube_quat, quat_threshold):
    """
    Determines if the cube is within quat_threshold of a xyz-aligned orientation, which means that
    one of the rotatable faces of the **full** cube is on top.
    This means that one of the axes of local coordinate system of the cube is pointing straight up.
    """
    z_up = np.array([0, 0, 1]).reshape(3, 1)
    mtx = quat2mat(cube_quat)
    # Axis that is the closest (by dotproduct) to z-up
    axis_nr = np.abs((z_up.T @ mtx)).argmax()

    # Axis of the cube pointing the closest to the top
    axis = mtx[:, axis_nr]
    axis = axis * np.sign(axis @ z_up)

    # Quaternion representing the rotation from "axis" that is almost up to
    # the actual "up" direction
    difference_quat = vectors2quat(axis, z_up[:, 0])

    return quat_magnitude(difference_quat) < quat_threshold


def random_unity2(random_state):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    """
    phi = random_state.uniform(0, np.pi * 2)
    costheta = random_state.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


from typing import Dict, List

import numpy as np

FINGERTIP_SITE_NAMES = [
    "robot0:S_fftip",
    "robot0:S_mftip",
    "robot0:S_rftip",
    "robot0:S_lftip",
    "robot0:S_thtip",
]

JOINTS = [
    "robot0:WRJ1",  # joint_id 00, actuator_id 00
    "robot0:WRJ0",  # joint_id 01, actuator_id 01
    "robot0:FFJ3",  # joint_id 02, actuator_id 02
    "robot0:FFJ2",  # joint_id 03, actuator_id 03
    "robot0:FFJ1",  # joint_id 04, actuator_id 04, tendon "FFT1", coupled joint
    "robot0:FFJ0",  # joint_id 05, actuator_id 04, tendon "FFT1", coupled joint
    "robot0:MFJ3",  # joint_id 06, actuator_id 05
    "robot0:MFJ2",  # joint_id 07, actuator_id 06
    "robot0:MFJ1",  # joint_id 08, actuator_id 07, tendon "MFT1", coupled joint
    "robot0:MFJ0",  # joint_id 09, actuator_id 07, tendon "MFT1", coupled joint
    "robot0:RFJ3",  # joint_id 10, actuator_id 08
    "robot0:RFJ2",  # joint_id 11, actuator_id 09
    "robot0:RFJ1",  # joint_id 12, actuator_id 10, tendon "RFT1", coupled joint
    "robot0:RFJ0",  # joint_id 13, actuator_id 10, tendon "RFT1", coupled joint
    "robot0:LFJ4",  # joint_id 14, actuator_id 11
    "robot0:LFJ3",  # joint_id 15, actuator_id 12
    "robot0:LFJ2",  # joint_id 16, actuator_id 13
    "robot0:LFJ1",  # joint_id 17, actuator_id 14, tendon "LFT1", coupled joint
    "robot0:LFJ0",  # joint_id 18, actuator_id 14, tendon "LFT1", coupled joint
    "robot0:THJ4",  # joint_id 19, actuator_id 15
    "robot0:THJ3",  # joint_id 20, actuator_id 16
    "robot0:THJ2",  # joint_id 21, actuator_id 17
    "robot0:THJ1",  # joint_id 22, actuator_id 18
    "robot0:THJ0",  # joint_id 23, actuator_id 19
]

DEFAULT_INITIAL_QPOS = {
    "robot0:WRJ1": -0.16514339750464327,
    "robot0:WRJ0": -0.31973286565062153,
    "robot0:FFJ3": 0.14340512546557435,
    "robot0:FFJ2": 0.32028208333591573,
    "robot0:FFJ1": 0.7126053607727917,
    "robot0:FFJ0": 0.6705281001412586,
    "robot0:MFJ3": 0.000246444303701037,
    "robot0:MFJ2": 0.3152655251085491,
    "robot0:MFJ1": 0.7659800313729842,
    "robot0:MFJ0": 0.7323156897425923,
    "robot0:RFJ3": 0.00038520700007378114,
    "robot0:RFJ2": 0.36743546201985233,
    "robot0:RFJ1": 0.7119514095008576,
    "robot0:RFJ0": 0.6699446327514138,
    "robot0:LFJ4": 0.0525442258033891,
    "robot0:LFJ3": -0.13615534724474673,
    "robot0:LFJ2": 0.39872030433433003,
    "robot0:LFJ1": 0.7415570009679252,
    "robot0:LFJ0": 0.704096378652974,
    "robot0:THJ4": 0.003673823825070126,
    "robot0:THJ3": 0.5506291436028695,
    "robot0:THJ2": -0.014515151997119306,
    "robot0:THJ1": -0.0015229223564485414,
    "robot0:THJ0": -0.7894883021600622,
}

ACTUATOR_CTRLRANGE = {
    "robot0:A_WRJ1": [-0.4887, 0.1396],  # DEGREES (-28, 8)
    "robot0:A_WRJ0": [-0.6981, 0.4887],  # DEGREES (-40, 28)
    "robot0:A_FFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "robot0:A_FFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "robot0:A_FFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "robot0:A_MFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "robot0:A_MFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "robot0:A_MFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "robot0:A_RFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "robot0:A_RFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "robot0:A_RFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "robot0:A_LFJ4": [0.0, 0.7854],  # DEGREES (0, 45)
    "robot0:A_LFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "robot0:A_LFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "robot0:A_LFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "robot0:A_THJ4": [-1.0472, 1.0472],  # DEGREES (-60, 60)
    "robot0:A_THJ3": [0.0, 1.2217],  # DEGREES (0, 70)
    "robot0:A_THJ2": [-0.2094, 0.2094],  # DEGREES (-12, 12)
    "robot0:A_THJ1": [-0.5236, 0.5236],  # DEGREES (-30, 30)
    "robot0:A_THJ0": [-1.5708, 0.0],  # DEGREES (-90, 0)
}

ACTUATOR_JOINT_MAPPING: Dict[str, List[str]] = {
    "robot0:A_WRJ1": ["robot0:WRJ1"],
    "robot0:A_WRJ0": ["robot0:WRJ0"],
    "robot0:A_FFJ3": ["robot0:FFJ3"],
    "robot0:A_FFJ2": ["robot0:FFJ2"],
    "robot0:A_FFJ1": ["robot0:FFJ1", "robot0:FFJ0"],  # Coupled joints
    "robot0:A_MFJ3": ["robot0:MFJ3"],
    "robot0:A_MFJ2": ["robot0:MFJ2"],
    "robot0:A_MFJ1": ["robot0:MFJ1", "robot0:MFJ0"],  # Coupled joints
    "robot0:A_RFJ3": ["robot0:RFJ3"],
    "robot0:A_RFJ2": ["robot0:RFJ2"],
    "robot0:A_RFJ1": ["robot0:RFJ1", "robot0:RFJ0"],  # Coupled joints
    "robot0:A_LFJ4": ["robot0:LFJ4"],
    "robot0:A_LFJ3": ["robot0:LFJ3"],
    "robot0:A_LFJ2": ["robot0:LFJ2"],
    "robot0:A_LFJ1": ["robot0:LFJ1", "robot0:LFJ0"],  # Coupled joints
    "robot0:A_THJ4": ["robot0:THJ4"],
    "robot0:A_THJ3": ["robot0:THJ3"],
    "robot0:A_THJ2": ["robot0:THJ2"],
    "robot0:A_THJ1": ["robot0:THJ1"],
    "robot0:A_THJ0": ["robot0:THJ0"],
}

FINGER_GEOM_MAPPING: Dict[str, List[int]] = {
    "PM": [0, 1, 2],
    "FF": [3, 4, 5],
    "MF": [6, 7, 8],
    "RF": [9, 10, 11],
    "LF": [12, 13, 14, 15],
    "TH": [16, 17, 18],
}

JOINT_LIMITS: Dict[str, np.ndarray] = {}
for actuator, ctrlrange in ACTUATOR_CTRLRANGE.items():
    joints = ACTUATOR_JOINT_MAPPING[actuator]
    for joint in joints:
        JOINT_LIMITS[joint] = np.array(ctrlrange) / len(joints)

MODES = {
    "face_up": [-np.pi / 2, 0, 0],
    "face_down": [np.pi / 2, 0, np.pi]
}


def get_actuator_mapping():
    def find(a, x):
        for idx, i in enumerate(a):
            if i == x:
                return idx
        raise Exception

    actuator_mapping = [None for i in JOINTS]
    for actuator in ACTUATOR_JOINT_MAPPING:
        for joint in ACTUATOR_JOINT_MAPPING[actuator]:
            actuator_mapping[find(JOINTS, joint)] = find(ACTUATOR_JOINT_MAPPING, actuator)

    return actuator_mapping


import os.path

import numpy as np
import xml.etree.ElementTree as et

from typing import Optional

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
XML_DIR = os.path.join(ASSETS_DIR)


def get_template(mjcf_xml):
    # no recursion - only one expansion is needed
    default_template = dict()
    for class_tempalte in mjcf_xml.root_element.find("default").findall("default"):

        template_name = class_tempalte.attrib.get("class", "")
        assert template_name, "class name cannot be empty!"

        default_template[template_name] = x_dict = dict()
        for body_attrib_template in class_tempalte:
            attrib_name = body_attrib_template.tag
            x_dict[attrib_name] = body_attrib_template.attrib

    return default_template


def combine_str(robot_name, part_name):
    return f"{robot_name}:{part_name}"


def get_mesh_data(rt: et.Element, name):
    asset_dir = os.path.join(ASSETS_DIR, "stls", "hand")
    for x in rt.findall(".//mesh"):
        if x.attrib.get("name", "") == name:
            mesh_file = x.attrib.get("file", "")
            mesh_file = os.path.join(asset_dir, mesh_file)

            scale = x.attrib.get("scale", "")

            # NOTE: ignore do not load mesh
            # assert os.path.exists(mesh_file)

            return mesh_file, scale


# credit: robogym
class MujocoXML:
    """
    Class that combines multiple MuJoCo XML files into a single one.
    """

    ###############################################################################################
    # CONSTRUCTION
    @classmethod
    def parse(cls, xml_filename: str, use_template=False):
        """ Parse given xml filename into the MujocoXML model """

        xml_full_path = os.path.join(XML_DIR, xml_filename)
        if not os.path.exists(xml_full_path):
            raise Exception(xml_full_path)

        with open(xml_full_path) as f:
            xml_root = et.parse(f).getroot()

        xml = cls(xml_root)
        xml.load_includes(os.path.dirname(os.path.abspath(xml_full_path)))
        if use_template:
            xml.apply_default_template()

        return xml

    @classmethod
    def from_string(cls, contents: str):
        """ Construct MujocoXML from string """
        xml_root = et.XML(contents)
        xml = cls(xml_root)
        xml.load_includes()
        return xml

    def __init__(self, root_element: Optional[et.Element] = None):
        """ Create new MujocoXML class """
        # This is the root element of the XML document we'll be modifying
        if root_element is None:
            # Create empty root element
            self.root_element = et.Element("mujoco")
        else:
            # Initialize it from the existing thing
            self.root_element = root_element

    def xml_string(self):
        """ Return combined XML as a string """
        return et.tostring(self.root_element, encoding="unicode", method="xml")

    def load_includes(self, include_root=""):
        """
        Some mujoco files contain includes that need to be process on our side of the system
        Find all elements that have an 'include' child
        """
        for element in self.root_element.findall(".//include/.."):
            # Remove in a second pass to avoid modifying list while iterating it
            elements_to_remove_insert = []

            for idx, subelement in enumerate(element):
                if subelement.tag == "include":
                    # Branch off initial filename
                    include_path = os.path.join(include_root, subelement.get("file"))

                    include_element = MujocoXML.parse(include_path)

                    elements_to_remove_insert.append(
                        (idx, subelement, include_element.root_element)
                    )

            # Iterate in reversed order to make sure indices are not screwed up
            for idx, to_remove, to_insert in reversed(elements_to_remove_insert):
                element.remove(to_remove)
                to_insert_list = list(to_insert)

                # Insert multiple elements
                for i in range(len(to_insert)):
                    element.insert(idx + i, to_insert_list[i])

        return self

    def apply_default_template(self):
        xml_root = self.root_element

        def get_template(rt: et.Element):
            # no recursion - only one expansion is needed
            default_template = dict()
            for class_template in rt.find("default").findall("default"):

                template_name = class_template.attrib.get("class", "")
                assert template_name, "class name cannot be empty!"

                default_template[template_name] = x_dict = dict()
                for body_attrib_template in class_template:
                    attrib_name = body_attrib_template.tag
                    x_dict[attrib_name] = body_attrib_template.attrib

            return default_template

        default_template = get_template(xml_root)

        def apply_template_to_attribs(template, x):
            for key in template:
                if key not in x.attrib.keys():
                    # apply k, v from template if k does not exist
                    x.set(key, template[key])

        def traverse(rt: et.Element):
            for x in rt:
                x_attribs = x.attrib
                if "class" in x_attribs:
                    template_name = x_attribs.get("class", "")
                    template = default_template[template_name]

                    apply_template_to_attribs(template[x.tag], x)


                elif "childclass" in x_attribs:

                    template_name = x_attribs.get("childclass", "")
                    template = default_template[template_name]

                    def apply_child_template(rt: et.Element):

                        for x in rt:
                            if x.tag in template:
                                apply_template_to_attribs(template[x.tag], x)

                            apply_child_template(x)

                traverse(x)

        rt = xml_root.find("worldbody")
        traverse(rt)


def homogeneous_matrix_from_pos_mat_np(pos, mat):
    m = np.eye(4)
    m[:3, :3] = mat
    m[:3, 3] = pos
    return m


def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape[-1] == 3
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.reshape(angle, axis[..., :1].shape)
    w = np.cos(angle / 2.0)
    v = np.sin(angle / 2.0) * axis
    quat = np.concatenate([w, v], axis=-1)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)

    assert np.array_equal(quat.shape[:-1], axis.shape[:-1])
    return quat


def get_joint_matrix(pos, angle, axis):
    def transform_rot_x_matrix(pos, angle):
        """
        Optimization - create a homogeneous matrix where rotation submatrix
        rotates around the X axis by given angle in radians
        """
        m = np.eye(4)
        m[1, 1] = m[2, 2] = np.cos(angle)
        s = np.sin(angle)
        m[1, 2] = -s
        m[2, 1] = s
        m[:3, 3] = pos
        return m

    def transform_rot_y_matrix(pos, angle):
        """
        Optimization - create a homogeneous matrix where rotation submatrix
        rotates around the Y axis by given angle in radians
        """
        m = np.eye(4)
        m[0, 0] = m[2, 2] = np.cos(angle)
        s = np.sin(angle)
        m[0, 2] = s
        m[2, 0] = -s
        m[:3, 3] = pos
        return m

    def transform_rot_z_matrix(pos, angle):
        """
        Optimization - create a homogeneous matrix where rotation submatrix
        rotates around the Z axis by given angle in radians
        """
        m = np.eye(4)
        m[0, 0] = m[1, 1] = np.cos(angle)
        s = np.sin(angle)
        m[0, 1] = -s
        m[1, 0] = s
        m[:3, 3] = pos
        return m

    if abs(axis[0]) == 1.0 and axis[1] == 0.0 and axis[2] == 0.0:
        return transform_rot_x_matrix(pos, angle * axis[0])
    elif axis[0] == 0.0 and abs(axis[1]) == 1.0 and axis[2] == 0.0:
        return transform_rot_y_matrix(pos, angle * axis[1])
    elif axis[0] == 0.0 and axis[1] == 0.0 and abs(axis[2]) == 1.0:
        return transform_rot_z_matrix(pos, angle * axis[2])
    else:
        return homogeneous_matrix_from_pos_mat_np(
            pos, quat2mat(quat_from_angle_and_axis(angle, axis))
        )


class MujocoGeom:
    def __init__(self,
                 parent_name: str,
                 matrix: np.array,
                 is_mesh: bool,
                 data,
                 ):
        self.parent_name = parent_name
        self.matrix = matrix
        self.is_mesh = is_mesh
        self.data = data

        # if mesh then we save the name
        # if not mesh then we save the type and size in a tuple


def prepare(filename, scale=1.0):
    mxml: MujocoXML = MujocoXML.parse(filename, use_template=True)
    root_body_name: str = "robot0:hand mount"
    root_body_pos: np.array = np.array([1.0, 1.25, 0.15])
    root_body_euler: np.array = np.array([np.pi / 2, 0, np.pi])
    target_sites: List[str] = FINGERTIP_SITE_NAMES
    joint_names: List[str] = JOINTS

    IDENTITY_QUAT = quat_identity()
    ROOT_BODY_PARENT = "NONE"

    target_sites_idx: Dict[str, int] = {
        v: idx for idx, v in enumerate(target_sites)
    }

    joint_names_idx: Dict[str, int] = {v: idx for idx, v in enumerate(joint_names)}
    num_sites = len(target_sites)
    site_info: List[Optional[Tuple]] = [None] * num_sites  # (4d matrix, parentBody)
    joint_info: List[Optional[Tuple]] = [None] * len(joint_names)  # (axis, pos)

    body_info: Dict[
        str, Any
    ] = dict()  # name => (4d homegeneous matrix, parentbody)
    body_joints: Dict[str, str] = dict()  # body => joints

    def get_matrix(x: et.Element):
        pos = np.fromstring(x.attrib.get("pos"), sep=" ") * scale
        if "euler" in x.attrib:
            euler = np.fromstring(x.attrib.get("euler"), sep=" ")
            return homogeneous_matrix_from_pos_mat_np(pos, euler2mat(euler))
        elif "axisangle" in x.attrib:
            axis_angle = np.fromstring(x.attrib.get("axisangle"), sep=" ")
            quat = quat_from_angle_and_axis(
                axis_angle[-1], np.array(axis_angle[:-1])
            )
            return homogeneous_matrix_from_pos_mat_np(pos, quat2mat(quat))
        elif "quat" in x.attrib:
            quat = np.fromstring(x.attrib.get("quat"), sep=" ")
            return homogeneous_matrix_from_pos_mat_np(pos, quat2mat(quat))
        else:
            quat = IDENTITY_QUAT
            return homogeneous_matrix_from_pos_mat_np(pos, quat2mat(quat))

    vis_geoms: List[MujocoGeom] = []
    col_geoms: List[MujocoGeom] = []

    def traverse(rt: et.Element, parent_body: str):
        assert rt.tag == "body", "only start from body tag in xml"
        matrix = get_matrix(rt)
        body_name = rt.attrib.get("name", "noname_body_%d" % len(body_info))
        body_info[body_name] = (matrix, parent_body)

        # parse joint
        x = rt.find("joint")
        joint_name = None
        if x is not None:
            joint_name = x.attrib.get("name", "")
            joint_idx: int = joint_names_idx.get(joint_name, -1)
            if joint_idx != -1:  # if in our target joints
                assert (
                        x.attrib.get("type", "hinge") == "hinge"
                ), "currently only support hinge joints"

                pos = np.fromstring(x.attrib.get("pos"), sep=" ")
                axis = np.fromstring(x.attrib.get("axis"), sep=" ")

                joint_info[joint_idx] = (pos, axis)

                assert (
                        joint_name not in body_joints
                ), "Only support open chain system, unsupported rigid bodies"
                body_joints[body_name] = joint_name

        # parse geometry
        for x in rt.findall("geom"):
            # get parent name
            parent_name = joint_name if joint_name is not None else body_name

            # get homogeneous matrix
            if not x.attrib.get("pos", ""):
                matrix = np.eye(4)
            else:
                matrix = get_matrix(x)

            # load data depending on whether it is mesh
            mesh_name = x.attrib.get("mesh", "")
            is_mesh = len(mesh_name) > 0
            if is_mesh:
                mesh_path, mesh_scale = get_mesh_data(mxml.root_element, mesh_name)
                data = {"path": mesh_path, "scale": mesh_scale}
            else:
                data = {k: x.attrib.get(k, "") for k in ("type", "size")}
                size = [float(x) * scale for x in data["size"].split(" ")]
                data["size"] = size

            geom = MujocoGeom(parent_name, matrix, is_mesh, data)

            # determine group
            geom_group = x.attrib.get("group", "")
            if geom_group == "1":
                vis_geoms.append(geom)
            elif geom_group == "4":
                col_geoms.append(geom)

        for x in rt.findall("site"):
            site_idx = target_sites_idx.get(x.attrib.get("name", ""), -1)
            if site_idx != -1:
                matrix = get_matrix(x)
                site_info[site_idx] = (matrix, body_name)

        for x in rt.findall("body"):
            traverse(x, body_name)

    #######################
    ### Begin traversal ###
    #######################

    rt = None
    for child in mxml.root_element.find("worldbody").findall("body"):  # type: ignore
        if child.attrib.get("name", "") == root_body_name:
            rt = child
            break

    assert rt is not None, "no root body found in xml"
    traverse(rt, ROOT_BODY_PARENT)

    ##########################
    # Build Computation Flow #
    ##########################

    root_matrix = homogeneous_matrix_from_pos_mat_np(
        root_body_pos, euler2mat(root_body_euler)
    )

    site_computations_inds = [[] for i in range(num_sites)]
    site_computations_mats = [[] for i in range(num_sites)]

    for i in range(num_sites):
        (matrix, parent_body) = site_info[i]
        while parent_body != ROOT_BODY_PARENT:
            parent_matrix, new_parent_body = body_info[parent_body]
            joint_name = body_joints.get(parent_body, "")

            # certain body has joint, certain doesn't
            if joint_name:
                site_computations_inds[i].append(("body_idx", len(site_computations_mats[i])))
                site_computations_mats[i].append(matrix)
                site_computations_inds[i].append(("joint_idx", joint_names_idx[joint_name]))
                matrix = parent_matrix
            else:
                matrix = parent_matrix @ matrix
            parent_body = new_parent_body

    site_computations = {
        "mats": site_computations_mats,
        "inds": [list(reversed(x)) for x in site_computations_inds],
    }

    def body_to_matrix(body_name):
        rt = None
        for child in mxml.root_element.find("worldbody").findall(".//body"):
            if child.attrib.get("name", "") == body_name:
                rt = child
                break
        return get_matrix(rt)

    root_info = {"root_matrix": root_matrix}
    for name in ["robot0:hand mount", "robot0:forearm", "robot0:wrist"]:
        root_info[name] = body_to_matrix(name)

    geometries: Dict[str, MujocoGeom] = {"vis": vis_geoms, "col": col_geoms}

    return root_info, site_computations, joint_info, geometries


def main():
    mjcf_xml = MujocoXML.parse("shadow/right_hand/shadow_hand.xml", use_template=True)

    root_info, site_computations, joint_info, geometries = prepare(mjcf_xml,
                                                                   "robot0:hand mount",
                                                                   [1.0, 1.25, 0.15],
                                                                   [np.pi / 2, 0, np.pi],
                                                                   FINGERTIP_SITE_NAMES,
                                                                   JOINTS,
                                                                   scale=1.0)

    # we need arm frame, if we want to use the arm collision geometry
    arm_frame = root_info["root_matrix"] @ root_info["robot0:hand mount"]

    # this is the actual root frame for the wrist and fingers
    root_frame = arm_frame @ root_info["robot0:wrist"]
    print(root_frame)
    exit(0)

    ################################
    ############ Chain Fk ##########
    ################################

    num_sites = 5
    num_joints = len(joint_info)
    max_timesteps = 1024

    joint_frame = np.zeros(shape=(max_timesteps, num_joints, 4, 4))
    joint_qpos = np.zeros(shape=(max_timesteps, num_joints,))

    def forward_kinematics(f):
        # compute kinematics at step f (not f + 1 !)

        for i in range(num_sites):

            m = root_frame

            # depending on idx_type , index could either be for body mats, or for joint index
            for idx_type, index in site_computations["inds"][i]:

                # it is NOT a joint index
                if idx_type != "joint_idx":
                    m = m @ site_computations["mats"][i][index]

                # it is a joint index
                else:
                    # read current joint angle
                    angle = joint_qpos[f, index]
                    pos, axis = joint_info[index]

                    joint_matrix = get_joint_matrix(pos, angle, axis)

                    m = m @ joint_matrix

                    joint_frame[f, index] = m

    ############################################
    ############ Collision geometries ##########
    ############################################

    forward_kinematics(0)

    SCALE = 1.0

    shape_primitives = {}
    for geom_type in ("box", "capsule"):
        shape_primitives[geom_type] = {x: [] for x in ("parent_frame", "joint_inds", "homogeneous_tfs", "sizes")}

    for geom in geometries["col"]:
        joint_idx = None
        if "forearm" in geom.parent_name:
            parent_frame = arm_frame
        else:
            joint_idx = JOINTS.index(geom.parent_name)
            parent_frame = joint_frame[joint_idx]

        if geom.is_mesh:
            continue
            # prim = MeshPrimitive(parent_frame, geom.matrix, geom.data["path"], geom.data["scale"], max_timesteps=max_timesteps, dtype=dtype)
            # self.primitives.append(prim)
        else:
            size = tuple(float(x) * SCALE for x in geom.data["size"].split(" "))
            shape_primitives[geom.data["type"]]["parent_frame"].append(parent_frame)
            shape_primitives[geom.data["type"]]["homogeneous_tfs"].append(geom.matrix)
            shape_primitives[geom.data["type"]]["sizes"].append(size)


if __name__ == '__main__':
    main()
