"""Calibration.py 数据标定说明

在稳定的场景下进行标定，确保 相机 和 激光雷达 正常稳定工作。

一般来说，数据量越多，标定结果越准确。
"""

from pathlib import Path

from base.action import dataset_action
from base.args_parser import DatasetArgsParser
from base.calibrate import calibrate_group, calibrate_unit
from base.datatype import NavioDataset, UnitData


def calibrate_dataset(path: Path | str, regen: bool = False):
    path = Path(path)
    ds = NavioDataset(path)

    def action(ud: UnitData):
        if regen or not ud.calibr_path.exists():
            calibrate_unit(ud, using_rerun=False)

    res = dataset_action(ds, action)

    if len(res) > 0:
        print("标定失败：")
        for base_dir, error in res:
            print(f"{base_dir}: {error}")


def main():
    # 读取命令行
    args = DatasetArgsParser()
    args.parser.add_argument(
        "--no_using_cam", action="store_true", help="标定时不使用相机数据"
    )
    args.parse()
    regen = args.regen

    if args.unit is not None:
        ud = UnitData(args.unit)
        calibrate_unit(ud, no_group=True, using_cam=not args.args.no_using_cam)
    elif args.group is not None:
        calibrate_group(args.group)
    elif args.dataset is not None:
        calibrate_dataset(args.dataset, regen=regen)


if __name__ == "__main__":
    main()
