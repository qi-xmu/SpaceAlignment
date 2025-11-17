"""Calibration.py 数据标定说明

在稳定的场景下进行标定，确保 相机 和 激光雷达 正常稳定工作。

一般来说，数据量越多，标定结果越准确。
"""

import argparse

from hand_eye import calibrate_group, calibrate_unit


def main():
    default_path = "dataset/001/20251031_01_in/Calibration/20251031_095725_SM-G9900"  # "dataset/20251111_204152_SM-G9900"
    # 读取命令行
    parser = argparse.ArgumentParser(description="Calibration")
    parser.add_argument("-d", "--dataset", help="Dataset path", required=False)
    parser.add_argument("-g", "--group", help="Group path", required=False)
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = default_path
        calibrate_unit(args.dataset)
    elif args.group is not None:
        calibrate_group(args.group)


if __name__ == "__main__":
    main()
