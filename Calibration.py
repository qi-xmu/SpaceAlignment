"""Calibration.py 数据标定说明

在稳定的场景下进行标定，确保 相机 和 激光雷达 正常稳定工作。

一般来说，数据量越多，标定结果越准确。
"""

import argparse

from base import Dataset
from hand_eye import calibrate_group, calibrate_unit


def calibrate_dataset(path: str):
    dataset = Dataset(path)
    for person in dataset.persons:
        for group in person.groups:
            calibrate_group(group)


def main():
    default_path = "dataset/001/20251031_01_in/Calibration/20251031_095725_SM-G9900"  # "dataset/20251111_204152_SM-G9900"
    # 读取命令行
    parser = argparse.ArgumentParser(description="Calibration")
    parser.add_argument("-u", "--unit", help="Dataset unit path", required=False)
    parser.add_argument("-g", "--group", help="Group path", required=False)
    parser.add_argument("-d", "--dataset", help="Dataset path", required=False)
    args = parser.parse_args()
    unit = args.unit
    group = args.group
    dataset = args.dataset

    if unit is None:
        unit = default_path
        calibrate_unit(unit)
    elif group is not None:
        calibrate_group(group)
    elif dataset is not None:
        calibrate_dataset(dataset)
    else:
        print("No calibration target specified.")


if __name__ == "__main__":
    main()
