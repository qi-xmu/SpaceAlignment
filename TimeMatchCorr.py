import json
from hand_eye import calibrate_Rgc, evaluate_and_save
from base import UnitPath, ARCoreData, RTABData, IMUData
from scipy.spatial.transform import Rotation
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def get_angvels(
    t_s: list[int],
    Rs1: list[np.ndarray],
    step=1,
):
    """获取角速度列表"""
    n = len(Rs1)
    assert n >= 2, "At least two rotations are required"

    As: list[np.ndarray] = []
    Ts = []
    for i in range(0, n - step, step):
        Ri = Rs1[i]
        Rj = Rs1[i + step]

        R_ji = Rj.T @ Ri
        rotvec = Rotation.from_matrix(R_ji).as_rotvec()
        angle = np.linalg.norm(rotvec)
        ang_vel = angle / (t_s[i + step] - t_s[i])
        As.append(ang_vel)
        Ts.append(t_s[i])
    return As, Ts


def match_correlation(
    Rs1,  ts1, Rs2, ts2, dataset_id=None, min_idx=10, max_idx=120, save=True
):
    """使用互相关法匹配Rs1和Rs2"""
    rate1 = round(1e6 / np.diff(ts1).mean())
    rate2 = round(1e6 / np.diff(ts2).mean())
    print(f"rate1 = {rate1}, rate2 = {rate2}")

    Rs1 = Rs1[min_idx * rate1: max_idx * rate1]
    Rs2 = Rs2[min_idx * rate2: max_idx * rate2]

    rads1, ts1_r = get_angvels(ts1, Rs1, step=rate1)
    rads2, ts2_r = get_angvels(ts2, Rs2, step=rate2)

    def resample(ts, vals, new_ts):
        interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
        return interp(new_ts)

    new_ts = np.linspace(max(ts1_r[0], ts2_r[0]),
                         min(ts1_r[-1], ts2_r[-1]), 1000)
    seq1 = resample(ts1_r, rads1, new_ts)
    seq2 = resample(ts2_r, rads2, new_ts)

    corr = correlate(seq1, seq2, mode="full")
    lag_arr = np.arange(-len(seq1) + 1, len(seq1))
    lag = lag_arr[np.argmax(corr)]
    t21_us = lag * (new_ts[1] - new_ts[0])

    # save
    if save:
        json_data = {
            "dataset_id": dataset_id,
            "tnc_us": t21_us,
            "notes": "t_node + tnc = t_cam",
        }

        ids = dataset_id.split("/")
        _id = ids[len(ids) - 1]

        path = UnitPath(dataset_id).target("match_time.json")
        with open(path, "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"保存 互相关法 匹配结果文件: {path}")

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
    min_idx = 10
    max_idx = 120
    # 2025-728-143709 2025-806-143700
    dataset_id = "RedmiK30PRO/2025-728-135334"
    paths = UnitPath(dataset_id)

    arcore = ARCoreData(paths.cam_path, dataset_id=dataset_id)
    rtab = RTABData(paths.gt_path)
    imu = IMUData(paths.imu_path)

    arcore.draw(show=False)
    rtab.draw(show=False)

    t_s = imu.t_us_f0
    Rs_bs, _ = imu.matches_Rsts()

    t_n = rtab.node_t_us_f0
    Rs_wn, ts_wn = rtab.matches_node_Rsts()

    # 互相关时间计算
    t21_us, ts, seq1, seq2 = match_correlation(
        Rs_bs, t_s, Rs_wn, t_n, dataset_id, min_idx, max_idx
    )
    match_correlation_draw(t21_us, ts, seq1, seq2, show=False)
    print(f"最佳时间偏移量 = {t21_us * 1e-6:.3f} 秒")

    from TimeMatch import match, match_draw
    f_t_s, f_t_n = (t_s - t21_us), t_n
    matches = match(f_t_s, f_t_n)
    match_draw(f_t_s, f_t_n, matches, show=True)

    # 标定
    matches = matches[:200]
    Rs1, ts1 = imu.matches_Rsts(matches)
    Rs2, ts2 = rtab.matches_node_Rsts(matches, inverse=True)
    R12, t12 = calibrate_Rgc(Rs1, ts1, Rs2, ts2)
    evaluate_and_save(dataset_id, R12, t12, Rs1, ts1, Rs2, ts2, is_save=True)


if __name__ == "__main__":
    main()
