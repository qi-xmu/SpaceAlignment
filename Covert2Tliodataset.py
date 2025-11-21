#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from pprint import pprint

import json
from pathlib import Path

import numpy as np
import pandas as pd

from base import load_calibration_data
from base.action import dataset_action, dataset_action_pa
from base.args_parser import DatasetArgsParser
from base.datatype import (
    IMUData,
    NavioDataset,
    RTABData,
    TimePoseSeries,
    UnitData,
)
from base.interpolate import get_time_series
from base.space import transform_local
from time_diff import match_correlation


class TLIO:
    """
    TLIO 数据集中的列名
    """

    t = "#timestamp [ns]"
    c = "temperature [degC]"
    w = ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"]
    a = ["a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]


JSON_TEMP = {
    "columns_name(width)": [
        "ts_us(1)",
        "gyr_compensated_rotated_in_World(3)",
        "acc_compensated_rotated_in_World(3)",
        "qxyzw_World_Device(4)",
        "pos_World_Device(3)",
        "vel_World(3)",
    ],
    "num_rows": 0,
    "approximate_frequency_hz": 0.0,
    "t_start_us": 0.0,
    "t_end_us": 0.0,
}


class TargetPaths:
    def __init__(self, target_root: Path) -> None:
        target_root.mkdir(parents=True, exist_ok=True)

        self.target_root = target_root
        self.csv_file = target_root / "imu_samples0.csv"
        self.npy_file = target_root / "imu0_resampled.npy"
        self.json_file = target_root / "imu0_resampled_description.json"


def UnitCovert(
    unit: UnitData,
    target_root: Path,
    extra: dict | None = None,
    rate: float = 200.0,
    regen: bool = False,
):
    target_path = TargetPaths(target_root)
    imu_data = IMUData(unit.imu_path)
    gt_data = RTABData(unit.gt_path)
    cd = load_calibration_data(unit=unit)

    # 使用csv存储原始数据
    imu_samples = pd.DataFrame()
    imu_samples[TLIO.t] = imu_data.t_sys_us
    imu_samples[TLIO.c] = np.zeros_like(imu_data.t_us)
    imu_samples[TLIO.w] = imu_data.gyro
    imu_samples[TLIO.a] = imu_data.acce
    imu_samples.to_csv(target_path.csv_file, index=False)

    # TODO  计算 IMU 旋转到全局坐标系下的数值，以及真值对齐到 IMU 的旋转矩阵 T_WI.
    if not target_path.npy_file.exists() or regen:
        # 真值空间变换
        # Groundtruth Body -> Sensor Body
        qs, ps = transform_local(
            tf_local=cd.tf_gs_local,
            qs=gt_data.node_qs,
            ps=gt_data.node_ps,
        )
        cs_g = TimePoseSeries(t_us=gt_data.t_sys_us, qs=qs, ps=ps)
        qs, ps = None, None

        # 计算时间偏差
        cs_i = imu_data.get_time_pose_series()
        t_21_us = match_correlation(cs1=cs_i, cs2=cs_g)
        cs_g.t_us += t_21_us

        # 插值并变换IMU数据
        t_us = get_time_series([gt_data.t_sys_us, imu_data.t_sys_us], rate=rate)
        cs_g = cs_g.interpolate(t_us)
        imu_data.interpolate(t_us)
        imu_data.transform_to_world()

        # 数据存储 quaternion形式 xyzw
        qs = np.array([[q.x, q.y, q.z, q.w] for q in cs_g.qs])
        ps = cs_g.ps

        _dt_s = np.diff(t_us, axis=0).reshape(-1, 1) * 1e-6
        vs = np.diff(ps, axis=0) / _dt_s
        vs = np.vstack([vs[0], vs])

        # 拼接
        t_us = t_us.reshape(-1, 1)
        imu0_resampled = np.hstack([t_us, imu_data.gyro, imu_data.acce, qs, ps, vs])
        np.save(target_path.npy_file, imu0_resampled)

        # 生成 JSON 文件
        json_info = JSON_TEMP.copy()
        json_info.update(extra or {})
        json_info["approximate_frequency_hz"] = rate
        json_info["num_rows"] = int(imu0_resampled.shape[0])
        json_info["t_start_us"] = float(t_us[0][0])
        json_info["t_end_us"] = float(t_us[-1][0])
        with open(target_path.json_file, "w") as f:
            json.dump(json_info, f, indent=4)
            print(f"Save {json_info}")

    return gt_data.t_len_s


if __name__ == "__main__":
    args = DatasetArgsParser().parse()
    assert args.dataset is not None, "Dataset path is required"
    assert args.output is not None, "Output path is required"
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    regen = args.regen

    if not output_path.exists():
        output_path.mkdir(parents=True)

    ds = NavioDataset(dataset_path)
    t_len_all_s = 0.0

    def action(ud: UnitData):
        global t_len_all_s
        t_len_s = UnitCovert(
            ud, target_root=output_path.joinpath(ud.data_id), regen=regen
        )
        t_len_all_s += t_len_s

    res = dataset_action_pa(ds, action)

    if len(res) > 0:
        print("Error occurred during conversion:")
        for item in res:
            print(f"{item[0]}: {item[1]}")

    print(f"Done， 数据集总时间： {t_len_all_s / 60:.2f}分钟")
