import argparse
import json
from pathlib import Path

import numpy as np

from base import GroupData, IMUData, RTABData, UnitData
from base.arcore_data import ARCoreData
from time_diff import match_correlation


class DataChecker:
    def __init__(self, ud: UnitData):
        self.ud = ud
        self.imu_data = IMUData(ud.imu_path)
        self.gt_data = RTABData(ud.gt_path)
        if self.ud.using_cam:
            self.cam_data = ARCoreData(ud.cam_path)

    def check_all(self):
        properties = dir(self)
        check_method = [
            method
            for method in properties
            if method.startswith("check_") and method != "check_all"
        ]
        check_list = {
            "data_path": str(self.ud.base_dir),
            "device_name": self.ud.device_name,
        }
        for method in check_method:
            check_list[method] = getattr(self, method)()

        check_json_str = json.dumps(check_list, indent=4, ensure_ascii=False)
        save_path = self.ud.base_dir.joinpath("data_check.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(check_json_str)
        print(check_json_str)

    def check_data_rate(self):
        res = {}
        imu_rate = self.imu_data.rate
        gt_rate = self.gt_data.rate
        res["imu"] = imu_rate
        res["rtab"] = gt_rate

        if self.ud.using_cam:
            cam_rate = self.cam_data.rate
            res["cam"] = cam_rate
        return res

    def check_time_diff(self):
        res = {}
        cs1 = self.imu_data.get_time_pose_series()
        cs2 = self.gt_data.get_time_pose_series()

        t21_us = match_correlation(cs1, cs2, show=True)
        res["time_diff_21_us"] = t21_us
        res["note"] = "检测两个序列的时间偏移"
        return res

    def check_groundtruth_gap(self, *, max_gap_s=2):
        res = {}
        ts = self.gt_data.node_t_us
        # 时间差距
        ts_diff = np.diff(ts) * 1e-6
        mean_gap = np.mean(ts_diff)
        max_gap = np.max(ts_diff)
        # 查询 大于 max_gap_s 的 索引的所有下标
        idxs = np.where(ts_diff > max_gap_s)[0].tolist()
        ts_diff = ts_diff[idxs].tolist()

        if max_gap > max_gap_s and False:
            # 查找最大间隔出现的时间
            self.gt_data.draw(
                mark_idxs=(idxs, ts_diff),
                show=False,
                save_path=self.ud.target("Trajectory.png"),
            )
        res["max_gap"] = max_gap
        res["mean_gap"] = mean_gap
        res["gap_idxs"] = idxs
        res["gap_diff"] = ts_diff
        res["note"] = "检测时间间隔"
        return res


if __name__ == "__main__":
    default_dataset_path = "dataset/001/20251031_01_in/20251031_101025_SM-G9900"
    default_dataset_path = "dataset/001/20251031_01_in/20251031_102355_SM-G9900"
    default_dataset_path = "dataset/001/20251031_01_in/20251031_103441_SM-G9900"

    # 解析命令行参数，获取数据集路径
    arg_parser = argparse.ArgumentParser(description="Ground Truth Analysis")
    arg_parser.add_argument("-u", "--unit", help="Path to the dataset unit")
    arg_parser.add_argument("-g", "--group", help="Group name")
    args = arg_parser.parse_args()
    dataset: str = args.unit
    group = args.group if args.group else "dataset/001/20251031_01_in"

    if dataset:
        ud = UnitData(dataset)
        checker = DataChecker(ud)
        checker.check_groundtruth_gap()
        checker.check_time_diff()
    elif group:
        gd = GroupData(group)
        for ud in gd.data:
            checker = DataChecker(ud)
            checker.check_all()
            # checker.check_groundtruth_gap()
            # checker.check_time_diff()
            break
