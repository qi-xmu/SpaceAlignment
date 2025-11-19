from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from scipy.signal import correlate

from base.datatype import ARCoreData, IMUData, RTABData, Time, TimePoseSeries, UnitData
from base.interpolate import get_time_series


def get_angvels(
    t_us: Time,
    qs: list[Quaternion],
    step: int = 1,
):
    """获取角速度列表"""
    n = len(qs)
    step = max(int(step), 1)
    assert n >= 2, "At least two rotations are required"

    As: list = []
    Ts = []
    for i in range(0, n - step, step):
        q_ij = qs[i].inverse * qs[i + step]
        dt_s = (t_us[i + step] - t_us[i]) * 1e-6
        assert dt_s > 0, "Time difference must be positive"
        ang_vel = q_ij.angle / dt_s
        As.append(ang_vel)
        Ts.append(t_us[i])
    return As, Ts


def match_correlation(
    cs1: TimePoseSeries,
    cs2: TimePoseSeries,
    *,
    time_range=(1, 20),
    resolution=100,
    save_path: Path | None = None,
    show=False,
):
    """使用互相关法匹配Rs1和Rs2"""
    # 分辨率不能大于时间序列的采样率，否则没有插值的意义
    resolution = min(resolution, cs1.rate, cs2.rate)

    t_new_us = get_time_series([cs1.t_us, cs2.t_us], *time_range, rate=resolution)
    cs1 = cs1.interpolate(t_new_us)
    cs2 = cs2.interpolate(t_new_us)
    print(f"使用时间范围：{(cs1.t_us[-1] - cs1.t_us[0]) / 1e6} 秒")

    seq1, t1 = get_angvels(cs1.t_us, cs1.qs, step=1)
    seq2, t2 = get_angvels(cs2.t_us, cs2.qs, step=1)
    t_new_us = t1

    corr = correlate(seq1, seq2, mode="full")
    lag_arr = np.arange(-len(seq1) + 1, len(seq1))
    lag = lag_arr[np.argmax(corr)]
    t21_us = lag * (t_new_us[1] - t_new_us[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Best Time Offset: {:.3f}s".format(t21_us * 1e-6))
    ax.plot(t_new_us - t21_us, seq1, label="Seq1", alpha=0.5)
    ax.plot(t_new_us, seq2, label="Seq2", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.legend()
    ax.grid()
    if save_path:
        fig.savefig(save_path)
    if not show:
        plt.close(fig)

    return int(t21_us)


def main():
    # path = "dataset/20251111_204152_SM-G9900"
    # path = "dataset/001/20251031_01_in/Calibration/20251031_095725_SM-G9900"
    path = "dataset/20251111_204152_SM-G9900"
    paths = UnitData(path)

    arcore = ARCoreData(paths.cam_path, z_up=False)
    rtab = RTABData(paths.gt_path)
    imu = IMUData(paths.imu_path)

    arcore.draw(show=False)
    rtab.draw(show=False)

    cs1 = imu.get_time_pose_series()
    cs2 = rtab.get_time_pose_series()

    # 互相关时间计算

    t21_us = match_correlation(cs1, cs2, show=True)
    print(f"最佳时间偏移量 = {t21_us * 1e-6:.3f} 秒")
    plt.show()


if __name__ == "__main__":
    main()
