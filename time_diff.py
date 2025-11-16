import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.spatial.transform import Rotation

from base import ARCoreData, IMUData, RTABData, UnitPath
from base.datatype import TimePoseSeries


def get_angvels(
    t_s: list[int] | np.ndarray,
    qs: list[Quaternion],
    step=1,
):
    """获取角速度列表"""
    n = len(qs)
    assert n >= 2, "At least two rotations are required"

    As: list = []
    Ts = []
    for i in range(0, n - step, step):
        q_ij = qs[i].inverse * qs[i + step]
        ang_vel = q_ij.angle / (t_s[i + step] - t_s[i])
        As.append(ang_vel)
        Ts.append(t_s[i])
    return As, Ts


def match_correlation(
    cs1: TimePoseSeries, cs2: TimePoseSeries, min_idx=10, max_idx=200, save=True
):
    """使用互相关法匹配Rs1和Rs2"""
    ts1, qs1 = cs1.ts_us, cs1.qs
    ts2, qs2 = cs2.ts_us, cs2.qs

    rate1 = round(1e6 / np.diff(ts1).mean())
    rate2 = round(1e6 / np.diff(ts2).mean())
    print(f"序列1频率 = {rate1}, 序列2频率 = {rate2}")

    Rs1 = qs1[min_idx * rate1 : max_idx * rate1]
    Rs2 = qs2[min_idx * rate2 : max_idx * rate2]

    rads1, ts1_r = get_angvels(ts1, qs1, step=rate1)
    rads2, ts2_r = get_angvels(ts2, qs2, step=rate2)

    def resample(ts, vals, new_ts):
        interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
        return interp(new_ts)

    new_ts = np.linspace(max(ts1_r[0], ts2_r[0]), min(ts1_r[-1], ts2_r[-1]), 1000)
    seq1 = resample(ts1_r, rads1, new_ts)
    seq2 = resample(ts2_r, rads2, new_ts)

    corr = correlate(seq1, seq2, mode="full")
    lag_arr = np.arange(-len(seq1) + 1, len(seq1))
    lag = lag_arr[np.argmax(corr)]
    t21_us = lag * (new_ts[1] - new_ts[0])

    return t21_us, new_ts, seq1, seq2


def match_correlation_draw(t21_us, ts, seq1, seq2, show=True):
    # 绘制偏移后的结果
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts - t21_us, seq1, label="ARCore", alpha=0.5)
    ax.plot(ts, seq2, label="RTAB", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.legend()
    ax.grid()
    if show:
        plt.show()


def main():
    path = "dataset/20251111_204152_SM-G9900"
    path = "dataset/001/20251031_01_in/Calibration/20251031_095725_SM-G9900"
    path = "dataset/20251111_204152_SM-G9900"
    paths = UnitPath(path)

    arcore = ARCoreData(paths.cam_path)
    rtab = RTABData(paths.gt_path)
    imu = IMUData(paths.imu_path)

    arcore.draw(show=False)
    rtab.draw(show=False)

    cs1 = imu.get_time_pose_series()
    cs2 = rtab.get_time_pose_series()

    # 互相关时间计算
    t21_us, ts, seq1, seq2 = match_correlation(cs1, cs2)
    print(f"最佳时间偏移量 = {t21_us * 1e-6:.3f} 秒")

    match_correlation_draw(t21_us, ts, seq1, seq2, show=True)


if __name__ == "__main__":
    main()
