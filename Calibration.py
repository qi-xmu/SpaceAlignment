"""Calibration.py 数据标定说明

在稳定的场景下进行标定，确保 相机 和 激光雷达 正常稳定工作。

一般来说，数据量越多，标定结果越准确。
"""

from pyquaternion import Quaternion
from base import UnitPath, ARCoreData, RTABData, GroupData
from hand_eye import calibrate_b1_b2
import numpy as np
import argparse


def calibrate_unit(path):
    print(f"Calibrating {path}")
    fp = UnitPath(path)
    data1 = ARCoreData(fp.cam_path)
    data2 = RTABData(fp.gt_path)

    cs1 = data1.get_calibr_series()
    cs2 = data2.get_calibr_series()

    # 标定结果，保存到文件
    pose_body1_body2, pose_ref1_ref2, matches = calibrate_b1_b2(
        cs_ref1_body1=cs1, cs_ref2_body2=cs2,
        result_path=fp.group(f"Calibration_{fp.device_name}.json")
    )
    q_b1_b2 = Quaternion(matrix=pose_body1_body2[0])
    q_r1_r2 = Quaternion(matrix=pose_ref1_ref2[0])

    print("Pose_Sensor_Groundtrhth: \n",
          q_b1_b2.axis, q_b1_b2.angle * 180 / np.pi, "\n",
          pose_body1_body2[1].flatten())
    print("Pose_World_Sensor_Groundtrhth: \n",
          q_r1_r2.axis, q_r1_r2.angle * 180 / np.pi, "\n",
          pose_ref1_ref2[1].flatten())


def calibrate_group(path):
    gp = GroupData(path)

    for unit in gp.raw_calibr_path.iterdir():
        if unit.is_dir():
            calibrate_unit(unit)
    pass


def main():
    # 读取命令行
    parser = argparse.ArgumentParser(description="Calibration")
    parser.add_argument("-d", "--dataset", help="Dataset path", required=False)
    parser.add_argument("-g", "--group", help="Group path", required=False)
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = "dataset/001/20251031_01_in/Calibration/20251031_095725_SM-G9900"
        calibrate_unit(args.dataset)
    elif args.group is not None:
        calibrate_group(args.group)


if __name__ == "__main__":
    main()
