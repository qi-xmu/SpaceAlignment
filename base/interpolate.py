import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from .basetype import Time


def slerp_rotation(rots: Rotation, t_old_us: NDArray, t_new_us: NDArray) -> Rotation:
    assert np.unique(t_old_us).size == len(t_old_us), "t_old_us must be unique"

    assert len(rots) == len(t_old_us)
    slerp = Slerp(t_old_us, rots)
    return slerp(t_new_us)


def interpolate_vector3d(
    vec3d: list[np.ndarray] | np.ndarray, t_old_us: Time, t_new_us: Time
) -> np.ndarray:
    assert len(vec3d) == len(t_old_us)
    vec3d = np.array(vec3d)
    t_old_us = np.array(t_old_us)
    t_new_us = np.array(t_new_us)

    assert t_new_us[0] >= t_old_us[0] and t_new_us[-1] <= t_old_us[-1], (
        "t_new_us must be within t_old_us"
    )

    interp = interp1d(
        t_old_us,
        vec3d,
        axis=0,
        kind="cubic",
        fill_value="extrapolate",
    )
    vec3d_new = interp(t_new_us)
    assert len(vec3d_new) == len(t_new_us), (
        f"Length mismatch, {len(vec3d_new)} != {len(t_new_us)}"
    )
    return vec3d_new


def get_time_series(
    ts_us: list[Time],
    time_range: tuple = (None, None),
    *,
    rate: float = 200,
) -> Time:
    t_start_us = int(max([t[0] for t in ts_us]))
    t_end_us = int(min([t[-1] for t in ts_us]))
    interval = int(1e6 / rate)
    assert t_start_us < t_end_us, f"Should: {t_start_us} < {t_end_us}"
    t_us = np.arange(t_start_us, t_end_us, interval, dtype=np.int64)
    # 限制时间轴长度
    ts, te = time_range
    start_idx = 0 if ts is None else int(max(ts * rate, 0))
    end_idx = len(t_us) if te is None else int(min(te * rate, len(t_us)))
    t_us = t_us[start_idx:end_idx]

    assert t_us[0] >= t_start_us and t_us[-1] <= t_end_us, (
        f"{t_us[0]} < {t_start_us} and {t_us[-1]} > {t_end_us}"
    )
    return t_us
