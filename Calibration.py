"""Calibration.py 数据标定说明

在稳定的场景下进行标定，确保 相机 和 激光雷达 正常稳定工作。

一般来说，数据量越多，标定结果越准确。
"""

from base import Dataset
from base.args_parser import DatasetArgsParser
from hand_eye import calibrate_group, calibrate_unit


def calibrate_dataset(path: str):
    dataset = Dataset(path)
    for person in dataset.persons:
        for group in person.groups:
            calibrate_group(group)


def main():
    # 读取命令行
    args = DatasetArgsParser().parse()

    if args.unit is not None:
        calibrate_unit(args.unit)
    elif args.group is not None:
        calibrate_group(args.group)
    elif args.dataset is not None:
        calibrate_dataset(args.dataset)


if __name__ == "__main__":
    main()
