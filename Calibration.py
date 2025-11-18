"""Calibration.py 数据标定说明

在稳定的场景下进行标定，确保 相机 和 激光雷达 正常稳定工作。

一般来说，数据量越多，标定结果越准确。
"""

import os
from pathlib import Path

from base import Dataset
from base.args_parser import DatasetArgsParser
from base.datatype import UnitData
from hand_eye import calibrate_group, calibrate_unit


def calibrate_dataset(path: Path | str, regen: bool = False):
    path = Path(path)
    dataset = Dataset(path)
    res = []
    idx = 0
    for person in dataset.persons:
        for group in person.groups:
            for unit in group.units:
                idx += 1
                print(f"\n{idx} ...")
                try:
                    if regen or not unit.calibr_path.exists():
                        # unit.using_cam = False
                        calibrate_unit(unit, using_rerun=False)
                    else:
                        print(f"跳过已标定的单元：{unit.base_dir}")
                except Exception as e:
                    res.append((unit.base_dir, e))

    if len(res) > 0:
        print("标定失败：")
        for base_dir, error in res:
            print(f"{base_dir}: {error}")
            target_dir = path.parent / "Error"
            target_dir.mkdir(parents=True, exist_ok=True)
            cmd = f"mv {base_dir} {target_dir} && echo '{error}' > {target_dir / base_dir.name / 'error.log'}"
            print(cmd)
            os.system(cmd)


def main():
    # 读取命令行
    args = DatasetArgsParser().parse()
    regen = args.regen

    if args.unit is not None:
        ud = UnitData(args.unit)
        calibrate_unit(ud)
    elif args.group is not None:
        calibrate_group(args.group)
    elif args.dataset is not None:
        calibrate_dataset(args.dataset, regen=regen)


if __name__ == "__main__":
    main()
