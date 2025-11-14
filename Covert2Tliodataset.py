#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from pprint import pprint
from typing import Literal
import pandas as pd
import numpy as np
from pathlib import Path
import json

from pyquaternion import Quaternion
from base import FilePath, PERSON_LIST, FlattenUnitData,  IMUData, RTABData


class TLIO:
    """
        TLIO 数据集中的列名
    """
    t = '#timestamp [ns]'
    c = 'temperature [degC]'
    w = ['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']
    a = ['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']


JSON_TEMP = {
    "columns_name(width)": [
        "ts_us(1)",
        "gyr_compensated_rotated_in_World(3)",
        "acc_compensated_rotated_in_World(3)",
        "qxyzw_World_Device(4)",
        "pos_World_Device(3)",
        "vel_World(3)"
    ],
    "num_rows": 0,
    "approximate_frequency_hz": 0.0,
    "t_start_us": 0.0,
    "t_end_us": 0.0
}


class GroundtruthData(RTABData):
    def __init__(self, file_path) -> None:
        super().__init__(file_path)

    def upsample(self, ts_us: np.ndarray, type: Literal["pos", "node_quat", "opt_quat"]) -> np.ndarray:

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
        self.csv_file = target_root/"imu_samples0.csv"
        self.npy_file = target_root/"imu0_resampled.npy"
        self.json_file = target_root/"imu0_resampled_description.json"


def UnitCovert(unit: FlattenUnitData, target_root: Path, extra: dict | None = None):
    target_path = TargetPaths(target_root)
    imu_data = IMUData(unit.imu_path.as_posix())
    gt_data = GroundtruthData(unit.gt_path.as_posix())

    # R_bn, t_bn = unit.parse_calibr_file()
    # AHRS 对齐 IMU 数据，并计算 IMU 旋转到全局坐标系下的数值，以及真值对齐到 IMU 的旋转矩阵 T_WI.

    # 保存数据
    if not target_path.csv_file.exists():
        imu_samples = pd.DataFrame()
        imu_samples[TLIO.t] = imu_data.t_us
        imu_samples[TLIO.c] = np.zeros_like(imu_data.t_us)
        imu_samples[TLIO.w] = imu_data.gyro
        imu_samples[TLIO.a] = imu_data.acce
        imu_samples.to_csv(target_path.csv_file, index=False)

    # TODO  计算 IMU 旋转到全局坐标系下的数值，以及真值对齐到 IMU 的旋转矩阵 T_WI.
    if not target_path.npy_file.exists():
        t_us = imu_data.t_us.reshape(-1, 1)

        ahrs = imu_data.unit_ahrs
        ahrs_Rs = np.array([it.rotation_matrix for it in ahrs])

        # R_WI = np.array(map(cvt, gt_qs))
        acc_W = np.einsum('ijk,ik->ij', ahrs_Rs, imu_data.acce)
        gyr_W = np.einsum('ijk,ik->ij', ahrs_Rs, imu_data.gyro)
        q_W = gt_data.upsample(t_us, "node_quat")
        p_W = gt_data.upsample(t_us, "pos")

        dt_s = np.diff(t_us, axis=0) * 1e-6
        v_W = np.diff(p_W, axis=0) / dt_s
        v_W = np.vstack([v_W[0], v_W])
        freq = 1 / dt_s.mean()

        # 拼接
        imu0_resampled = np.hstack([t_us, gyr_W, acc_W, q_W, p_W, v_W])
        np.save(target_path.npy_file, imu0_resampled)

    if not target_path.json_file.exists():
        json_info = JSON_TEMP.copy()
        json_info.update(extra or {})
        json_info['approximate_frequency_hz'] = float(freq)
        json_info['num_rows'] = int(imu0_resampled.shape[0])
        json_info['t_start_us'] = float(t_us[0][0])
        json_info['t_end_us'] = float(t_us[-1][0])
    else:
        with open(target_path.json_file, "r") as f:
            json_info = json.load(f)
        json_info.update(extra or {})

    with open(target_path.json_file, "w") as f:
        json.dump(json_info, f, indent=4)
        print(f"Save {json_info}")


if __name__ == "__main__":
    fp = FilePath("./dataset", PERSON_LIST)
    flatten_datas = fp.flatten()
    UnitCovert(flatten_datas[0], target_root=Path("target"))
