import numpy as np
from pyquaternion import Quaternion

from base.datatype import Pose


def transform_world(
    *,
    tf_world: Pose,
    qs: list[Quaternion],
    ps: np.ndarray,
):
    world_rot, world_p = tf_world
    assert world_rot is not None and world_p is not None, "Pose must be valid"
    world_q = Quaternion(matrix=world_rot)

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
    local_q = Quaternion(matrix=local_rot)

    ps = np.array([q.rotate(local_p) for q in qs]) + ps
    qs = [q * local_q for q in qs]

    # ps = np.einsum("jk,ik->ij", local_rot, ps) + local_pos
    return qs, ps
