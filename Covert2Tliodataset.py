#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from pprint import pprint
import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from base import (
    Dataset,
    FlattenUnitData,
    IMUData,
    RTABData,
    TimePoseSeries,
)
from base.interpolate import get_time_series, pose_interpolate
from base.space import transform_local
from hand_eye import load_calibration_data


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


class GroundtruthData(RTABData):
    def __init__(self, file_path) -> None:
        super().__init__(file_path)

    def upsample(
        self, ts_us: np.ndarray, type: Literal["pos", "node_quat", "opt_quat"]
    ) -> np.ndarray:
        def _sample_quat(ts_us: np.ndarray, qs: list[Quaternion]) -> np.ndarray:
            qs_resampled = []
            q = Quaternion.slerp(qs[0], qs[1])

            qs_resampled.append(q)
            return np.array(qs_resampled)

        return np.array([])


class TargetPaths:
    def __init__(self, target_root: Path) -> None:
        target_root.mkdir(parents=True, exist_ok=True)

        self.target_root = target_root
        self.csv_file = target_root / "imu_samples0.csv"
        self.npy_file = target_root / "imu0_resampled.npy"
        self.json_file = target_root / "imu0_resampled_description.json"


def UnitCovert(
    unit: FlattenUnitData,
    target_root: Path,
    extra: dict | None = None,
    rate: float = 200.0,
    regen: bool = False,
):
    target_path = TargetPaths(target_root)
    imu_data = IMUData(unit.imu_path)
    gt_data = GroundtruthData(unit.gt_path)
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
        # 计算新数据
        qs, ps = transform_local(
            tf_local=cd.tf_local,
            qs=gt_data.node_qs,
            ps=gt_data.node_ps,
        )
        cs_g = TimePoseSeries(ts=gt_data.t_sys_us, qs=qs, ps=ps)
        qs, ps = None, None

        t_us = get_time_series([gt_data.t_sys_us, imu_data.t_sys_us], rate=rate)
        cs_g = pose_interpolate(cs=cs_g, t_new_us=t_us)
        imu_data.interpolate(t_us)
        imu_data.transform_to_world()

        qs = np.array([it.elements for it in cs_g.qs])
        ps = cs_g.ps

        _dt_s = np.diff(t_us, axis=0).reshape(-1, 1) * 1e-6
        vs = np.diff(ps, axis=0) / _dt_s
        vs = np.vstack([vs[0], vs])

        # # 拼接
        t_us = t_us.reshape(-1, 1)
        imu0_resampled = np.hstack([t_us, imu_data.gyro, imu_data.acce, qs, ps, vs])
        np.save(target_path.npy_file, imu0_resampled)

        if not target_path.json_file.exists():
            json_info = JSON_TEMP.copy()
            json_info.update(extra or {})
            json_info["approximate_frequency_hz"] = rate
            json_info["num_rows"] = int(imu0_resampled.shape[0])
            json_info["t_start_us"] = float(t_us[0][0])
            json_info["t_end_us"] = float(t_us[-1][0])
        else:
            with open(target_path.json_file, "r") as f:
                json_info = json.load(f)
            json_info.update(extra or {})

        with open(target_path.json_file, "w") as f:
            json.dump(json_info, f, indent=4)
            print(f"Save {json_info}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--dataset", type=str)
    arg_parser.add_argument("-t", "--target", type=str)

    args = arg_parser.parse_args()
    dataset_path = Path(args.dataset)
    target_path = Path(args.target)

    if not target_path.exists():
        target_path.mkdir(parents=True)

    fp = Dataset(dataset_path, ["001"])
    flatten_data = fp.flatten()
    for i, flatten0 in enumerate(flatten_data):
        print(i, "...")
        UnitCovert(flatten0, target_root=target_path.joinpath(flatten0.data_id))
    print("Done")
