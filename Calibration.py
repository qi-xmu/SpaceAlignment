"""Calibration.py 数据标定说明

在稳定的场景下进行标定，确保 相机 和 激光雷达 正常稳定工作。

一般来说，数据量越多，标定结果越准确。
"""

from base.action import dataset_action
from base.args_parser import DatasetArgsParser
from base.calibrate import calibrate_unit
from base.datatype import GroupData, NavioDataset, UnitData


def main():
    # 读取命令行
    args = DatasetArgsParser()
    args.parser.add_argument(
        "--no_using_cam", action="store_true", help="标定时不使用相机数据"
    )
    args.parse()
    regen = args.regen

    def action(ud: UnitData):
        if regen or not ud.calibr_path.exists():
            calibrate_unit(
                ud,
                using_rerun=args.visual,
                using_cam=not args.args.no_using_cam,
                z_up=args.z_up,
            )

    if args.unit is not None:
        ud = UnitData(args.unit)
        args.visual = True  # 单数据默认为真
        action(ud)
    elif args.group is not None:
        gp = GroupData(args.group)
        for unit in gp.calib_units:
            action(unit)
    elif args.dataset is not None:
        ds = NavioDataset(args.dataset)
        res = dataset_action(ds, action)
        if len(res) > 0:
            print("标定失败：")
            for base_dir, error in res:
                print(f"{base_dir}: {error}")


if __name__ == "__main__":
    main()
