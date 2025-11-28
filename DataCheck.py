#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

import numpy as np
from matplotlib import pyplot as plt

from base.action import dataset_action
from base.args_parser import DatasetArgsParser
from base.datatype import (
    ARCoreData,
    GroupData,
    IMUData,
    NavioDataset,
    RTABData,
    UnitData,
)
from base.time_diff import match_correlation


class DataChecker:
    def __init__(
        self,
        ud: UnitData,
        *,
        is_visual: bool = True,
        time_range: tuple[float | None, float | None] = (0, 50),
        regen: bool = False,
        z_up: bool = False,
    ):
        self.ud = ud
        self.imu_data = IMUData(ud.imu_path)
        self.gt_data = RTABData(ud.gt_path)
        if self.ud.using_cam:
            self.cam_data = ARCoreData(ud.cam_path, z_up=z_up)

        self.is_visual = is_visual
        self.regen = regen
        self.time_range = time_range

    def run_checks(self):
        if self.ud.check_file.exists() and not self.regen:
            print("检查文件已存在")
            return

        properties = dir(self)
        check_method = [method for method in properties if method.startswith("check_")]
        check_list = {
            "data_path": str(self.ud.base_dir),
            "device_name": self.ud.device_name,
        }
        for method in check_method:
            check_list[method] = getattr(self, method)()

        check_json_str = json.dumps(check_list, indent=4, ensure_ascii=False)
        save_path = self.ud.base_dir.joinpath("DataCheck.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
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

        t21_us = match_correlation(
            cs1,
            cs2,
            time_range=self.time_range,
            show=self.is_visual,
            save_path=self.ud.target("TimeDiff.png"),
        )
        res["time_diff_21_us"] = t21_us
        res["note"] = "检测两个序列的时间偏移"
        return res

    def check_groundtruth_gap(self, *, max_gap_s=None):
        res = {}
        ts = self.gt_data.t_sys_us
        # 时间差距
        ts_diff = np.diff(ts) * 1e-6
        mean_gap = np.mean(ts_diff)
        max_gap = np.max(ts_diff)
        # 查询 大于 max_gap_s 的 索引的所有下标
        max_gap_s = 1 if max_gap_s is None else max_gap_s
        idxs = np.where(ts_diff > max_gap_s)[0].tolist()
        ts_diff = ts_diff[idxs].tolist()

        print(list(zip(idxs, ts_diff)))

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
    # 解析命令行参数，获取数据集路径
    args = DatasetArgsParser()
    args.parse()
    unit_path = args.unit
    group_path = args.group
    dataset_path = args.dataset
    time_range = args.time_range
    visual = args.visual

    print(time_range)

    def action(ud):
        DataChecker(
            ud,
            is_visual=visual,
            time_range=time_range,
            z_up=args.z_up,
            regen=args.regen,
        ).run_checks()
        if visual:
            plt.show()

    if unit_path:
        ud = UnitData(unit_path)
        action(ud)
    elif group_path:
        gd = GroupData(group_path)
        dataset_action(gd, action)
    elif dataset_path:
        ds = NavioDataset(dataset_path)
        dataset_action(ds, action)
    print("Analysis completed.")
