import numpy as np
from numpy.typing import NDArray
from pyquaternion import Quaternion

from .basetype import Pose


def rotation_matrix_to_quaternion(R):
    """将3x3旋转矩阵转换为四元数 (w, x, y, z)"""
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    if qw < 0:
        qw = -qw
        qx = -qx
        qy = -qy
        qz = -qz

    return Quaternion([qw, qx, qy, qz])


def OrtR(R: NDArray):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt @ np.diag([1, 1, np.linalg.det(U @ Vt)])


def transform_world(
    *,
    tf_world: Pose,
    qs: list[Quaternion],
    ps: np.ndarray,
):
    world_rot, world_p = tf_world
    assert world_rot is not None and world_p is not None, "Pose must be valid"
    world_q = Quaternion(matrix=OrtR(world_rot))

    qs = [world_q * q for q in qs]
    ps = np.einsum("jk,ik->ij", world_rot, ps) + world_p
    return qs, ps


def transform_local(
    *,
    tf_local: Pose,
    qs: list[Quaternion],
    ps: np.ndarray,
):
    local_rot, local_p = tf_local
    assert local_rot is not None and local_p is not None, "Pose must be valid"
    local_q = Quaternion(matrix=OrtR(local_rot))

    ps = np.array([q.rotate(local_p) for q in qs]) + ps
    qs = [q * local_q for q in qs]

    # ps = np.einsum("jk,ik->ij", local_rot, ps) + local_pos
    return qs, ps
