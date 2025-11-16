import os

import matplotlib.pyplot as plt
import numpy as np

from base import ARCoreData, RTABData, UnitPath
from base.datatype import TimePoseSeries
from time_diff import match_correlation, match_correlation_draw


# 计算ARCore的时间间隔
def _cal_time_interval(arcore_times):
    """计算ARCore的时间间隔（频率）"""
    if len(arcore_times) < 2:
        return 0
    intervals = np.diff(arcore_times)
    return np.median(intervals)  # 使用中位数作为典型间隔


def find_closest_matches_with_threshold(ts1, ts2, max_time_diff=None):
    """
    为每个ARCore时间点找到最接近的RTAB时间点，使用时间差阈值

    Args:
        ts1: ARCore时间轴列表
        ts2: RTAB时间轴列表
        max_time_diff: 最大允许时间差，如果为None则使用ARCore间隔

    Returns:
        matches: 匹配对列表，每个元素包含(arcore_idx, rtab_idx, time_diff)
        unpaired_arcore: 未匹配的ARCore索引列表
        unpaired_rtab: 未匹配的RTAB索引列表
    """
    if max_time_diff is None:
        # 计算ARCore的时间间隔作为最大允许时间差
        inter_ts1 = _cal_time_interval(ts1)
        inter_ts2 = _cal_time_interval(ts2)
        max_time_diff = min(inter_ts1, inter_ts2) / 1.9

    matches = []
    used_indices2 = set()

    for idx1, t1 in enumerate(ts1):
        min_diff = float("inf")
        best_idx2 = -1

        # 找到最接近的RTAB时间点
        for idx2, t2 in enumerate(ts2):
            if idx2 in used_indices2:
                continue

            time_diff = abs(t1 - t2)
            if time_diff < min_diff:
                min_diff = time_diff
                best_idx2 = idx2

        # 检查时间差是否在允许范围内
        if best_idx2 >= 0 and min_diff <= max_time_diff:
            matches.append((idx1, best_idx2, min_diff))
            used_indices2.add(best_idx2)
        else:
            # 时间差太大，不匹配
            pass

    # 找出未匹配的索引
    unpaired_arcore = [
        i for i in range(len(ts1)) if i not in [match[0] for match in matches]
    ]
    unpaired_rtab = [
        i for i in range(len(ts2)) if i not in [match[1] for match in matches]
    ]

    return matches, unpaired_arcore, unpaired_rtab


def nearest_match(
    cs1: TimePoseSeries,
    cs2: TimePoseSeries,
    *,
    save_path=None,
    is_t_diff=True,
    show_t_diff=False,
):
    assert len(cs1) > 0 or len(cs2) > 0, "cs1 or cs2 is empty"

    ts1, ts2 = cs1.ts_us, cs2.ts_us
    t21_us = 0

    if is_t_diff:
        t21_us, ts, seq1, seq2 = match_correlation(cs1, cs2)
        ts2 = ts2 + t21_us
        print(f"匹配时间差: {t21_us * 1e-6:.6f} s")
        match_correlation_draw(t21_us, ts, seq1, seq2, show=show_t_diff)

    matches, _, _ = find_closest_matches_with_threshold(ts1, ts2)

    # 打印匹配结果统计
    print(f"成功匹配对数: {len(matches)}")
    return matches, t21_us


def match_draw(ts1, ts2, matches, show=True, save_path=None):
    time_diffs_ms = [match[2] / 1000 for match in matches]  # 转换为毫秒
    avg_diff = sum(time_diffs_ms) / len(time_diffs_ms)
    max_diff = max(time_diffs_ms)
    # 如果有匹配结果，绘制匹配连接图
    fig2 = plt.figure(figsize=(14, 8))
    ax2 = fig2.add_subplot(111)

    # 绘制原始数据点
    ax2.scatter(ts1, [1] * len(ts1), s=3, label="TS1", color="blue")
    ax2.scatter(ts2, [2] * len(ts2), s=3, label="TS2", color="red")

    # 获取所有时间差用于颜色映射（转换为毫秒）
    time_diffs_ms = [match[2] / 1000 for match in matches]  # 转换为毫秒
    norm = plt.Normalize(  # type: ignore
        min(time_diffs_ms),
        max(time_diffs_ms),
    )
    # 根据时间差大小设置颜色（使用毫秒），误差越小颜色越暗（紫色），误差越大颜色越亮（黄色）
    cmap = plt.cm.viridis  # type: ignore

    linewidth = 0.5
    for idx1, idx2, time_diff in matches:
        p1, p2 = ts1[idx1], ts2[idx2]
        color = cmap(norm(time_diff / 1000))
        ax2.plot([p1, p2], [1, 2], color=color, linewidth=linewidth)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.8)
    cbar.set_label("Time Difference (ms)", rotation=270, labelpad=15)

    ax2.set_xlabel("Time [μs]")
    ax2.set_ylabel("Data Source")
    ax2.set_title(f"ARCore vs RTAB Time Matching (showing {len(matches)} matches)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(["TimeSeries1", "TimeSeries2"])

    # 添加匹配统计信息
    match_stats = f"Matched pairs: {len(matches)}/{min(len(ts1), len(ts2))}\n"
    match_stats += f"Avg time diff: {avg_diff:.3f} ms\n"
    match_stats += f"Max time diff: {max_diff:.3f} ms"

    ax2.text(
        0.02,
        0.5,
        match_stats,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 设置中文字体支持
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/time_matching.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    path = "dataset/20251111_204152_SM-G9900"
    params = UnitPath(path)
    rtab_data = RTABData(params.gt_path)
    arcore_data = ARCoreData(params.cam_path)

    # 获取时间轴数据
    cs1 = arcore_data.get_time_pose_series()
    cs2 = rtab_data.get_time_pose_series()
    matches = nearest_match(cs1, cs2)
