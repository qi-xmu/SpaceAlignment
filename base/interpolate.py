import numpy as np
from pyquaternion import Quaternion
from scipy.interpolate import interp1d

from base.datatype import Time, TimePoseSeries


def slerp_quaternion(
    *,
    qs: list[Quaternion],
    t_old_us: Time,
    t_new_us: Time,
) -> list[Quaternion]:
    t_old_us = np.array(t_old_us)
    t_new_us = np.array(t_new_us)

    assert t_new_us[0] >= t_old_us[0] and t_new_us[-1] <= t_old_us[-1], (
        f"t: [{t_new_us[0] - t_old_us[0]} - {t_new_us[-1] - t_old_us[-1]}]"
    )

    n_old = len(t_old_us)
    n_new = len(t_new_us)
    qs_new = []
    ptr = 0
    for i in range(n_new):
        while t_new_us[i] > t_old_us[ptr] and ptr + 1 < n_old:
            ptr += 1
        # 寻找插值的左侧
        pre = max(ptr - 1, 0)
        while t_old_us[pre] > t_new_us[i] and pre > 0:
            pre = pre - 1

        if t_new_us[i] == t_old_us[ptr]:
            qs_new.append(qs[ptr])
        else:
            assert t_old_us[ptr] != t_old_us[pre], f"Duplicate time {t_old_us[ptr]}"
            amount = (t_new_us[i] - t_old_us[pre]) / (t_old_us[ptr] - t_old_us[pre])
            q = Quaternion.slerp(
                qs[pre],
                qs[ptr],
                amount,
            )
            qs_new.append(q)
    return qs_new


def interpolate_vector3d(
    *,
    vec3d: list[np.ndarray] | np.ndarray,
    t_old_us: Time,
    t_new_us: Time,
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
        kind="linear",
        fill_value="extrapolate",
    )
    vec3d_new = interp(t_new_us)
    assert len(vec3d_new) == len(t_new_us), (
        f"Length mismatch, {len(vec3d_new)} != {len(t_new_us)}"
    )
    return vec3d_new


def pose_interpolate(
    *,
    cs: TimePoseSeries,
    t_new_us: Time,
) -> TimePoseSeries:
    qs = slerp_quaternion(
        qs=cs.qs,
        t_old_us=cs.t_us,
        t_new_us=t_new_us,
    )
    ps = interpolate_vector3d(
        vec3d=cs.ps,
        t_old_us=cs.t_us,
        t_new_us=t_new_us,
    )
    return TimePoseSeries(ts=t_new_us, qs=qs, ps=ps)


def get_time_series(ts_us: list[Time], rate: float = 200.0) -> Time:
    t_start_us = max([t[0] for t in ts_us])
    t_end_us = min([t[-1] for t in ts_us])
    interval = 1e6 / rate
    t_us = np.arange(t_start_us + interval, t_end_us - interval, interval)
    assert t_us[0] >= t_start_us and t_us[-1] <= t_end_us, (
        "t_us must be within t_start_us and t_end_us"
    )
    return t_us
