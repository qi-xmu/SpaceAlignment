import argparse

import numpy as np

from base import IMUData, RTABData, UnitData
from base.datatype import GroupData
from time_diff import match_correlation


class DataChecker:
    def __init__(self, ud: UnitData):
        self.ud = ud
        self.imu_data = IMUData(ud.imu_path)
        self.gt_data = RTABData(ud.gt_path)

    def check_time_diff(self):
        cs1 = self.imu_data.get_time_pose_series()
        cs2 = self.gt_data.get_time_pose_series()

        t21_us = match_correlation(cs1, cs2, show=True)
        print(f"最佳时间偏移量 = {t21_us * 1e-6:.3f} 秒")
        return t21_us

    def check_groundtruth_gap(self, *, max_gap_s=2):
        ts = self.gt_data.node_t_us
        print(f"频率: {1e6 / np.mean(np.diff(ts))} Hz")
        # 时间差距
        ts_diff = np.diff(ts) * 1e-6
        mean_gap = np.mean(ts_diff)
        max_gap = np.max(ts_diff)
        # 查询 大于 max_gap_s 的 索引的所有下标
        idxs = np.where(ts_diff > max_gap_s)[0].tolist()
        ts_diff = ts_diff[idxs].tolist()
        print(ts_diff)

        print(f"时间间隔 最大/平均: {max_gap:.3f} s / {mean_gap:.3f} s")
        if max_gap > max_gap_s:
            # 查找最大间隔出现的时间
            max_idx = np.argmax(ts_diff)
            print(
                f"! 时间间隔过大，位置 {max_idx}:{ts[max_idx]} s，最大间隔 {max_gap:.3f} s"
            )
            print(f"! All = {idxs}, {ts_diff}")
            self.gt_data.draw(mark_idxs=(idxs, ts_diff), show=False)
            return idxs
        return []


if __name__ == "__main__":
    default_dataset_path = "dataset/001/20251031_01_in/20251031_101025_SM-G9900"
    default_dataset_path = "dataset/001/20251031_01_in/20251031_102355_SM-G9900"
    default_dataset_path = "dataset/001/20251031_01_in/20251031_103441_SM-G9900"

    # 解析命令行参数，获取数据集路径
    arg_parser = argparse.ArgumentParser(description="Ground Truth Analysis")
    arg_parser.add_argument("-d", "--dataset", help="Path to the dataset")
    arg_parser.add_argument("-g", "--group", help="Group name")
    args = arg_parser.parse_args()
    dataset: str = args.dataset
    group = args.group if args.group else "dataset/001/20251031_01_in"

    if group:
        gd = GroupData(group)
        for ud in gd.data:
            checker = DataChecker(ud)
            checker.check_groundtruth_gap()
            checker.check_time_diff()

    if dataset:
        ud = UnitData(dataset)
        checker = DataChecker(ud)
        checker.check_groundtruth_gap()
        checker.check_time_diff()
