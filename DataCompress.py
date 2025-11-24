"""
Module for compressing data.

Author: qi
License: MIT

目前存在的问题：
1. imu数据没有 t_sys_us 数据
2. 没有标定数据

Usage:
    python DataCompress.py -d <dataset_path> -o <output_path>
"""

import os
from pathlib import Path

from base.action import dataset_action, dataset_action_pa  # noqa
from base.args_parser import DatasetArgsParser
from base.datatype import ARCoreData, IMUData, NavioDataset, RTABData, UnitData

"""
数据读取。
默认格式：[BaseUrl]/[DeviceName]/[UnitData]/**
"""


class Target:
    """
    Target data for save path.

    [BaseUrl]/[Group_Scene]/[Unit_Device]
    """

    target_path: Path
    group_fmt: str
    scene_name: str

    def __init__(self, target_path: Path | str):
        self.target_path = Path(target_path)
        self.person = "001"  # 默认人员
        self.group_fmt = "2025_{}_{}"  # 默认组名称
        self.scene_name = "in"  # 默认场景 indoor

    def unit(self, data_id: str, device_name: str):
        group_name = self.group_fmt.format(device_name, self.scene_name)
        unit_path = (
            self.target_path.joinpath(self.person)
            .joinpath(group_name)
            .joinpath(data_id)
        )
        imu_path = unit_path.joinpath("imu.csv")
        cam_path = unit_path.joinpath("cam.csv")
        gt_path = unit_path.joinpath("rtab.csv")
        return (unit_path, imu_path, cam_path, gt_path)


class CompressUnitData(UnitData):
    def __init__(self, base_dir: str | Path, device_name: str):
        UnitData.__init__(self, base_dir)

        self.device_name = device_name  # type: ignore
        self.data_id = f"{self.data_id}_{self.device_name}"

    @classmethod
    def from_unit(cls, unit: UnitData):
        self = cls(unit.base_dir, unit.device_name)
        self.data_id = unit.data_id
        return self

    @staticmethod
    def copy_file(src: Path, dst: Path):
        """Copy file from src to dst."""
        print(f"Copying {src} to {dst}")
        dst.write_bytes(src.read_bytes())

    def compress(
        self,
        target: Target,
        *,
        regen: bool = False,
        using_cam: bool = True,
        using_opt: bool = False,
        is_z_up: bool = False,
    ):
        if self.err_msg:
            return
        t_base_us = 0
        target_unit_path, new_imu_path, new_cam_path, new_gt_path = target.unit(
            self.data_id, self.device_name
        )

        if not new_imu_path.exists() or regen:
            gt_data = RTABData(self.gt_path, is_load_opt=using_opt)
            target_unit_path.mkdir(parents=True, exist_ok=True)
            gt_data.save_csv(new_gt_path, using_opt=using_opt)
        else:
            gt_data = RTABData(new_gt_path)
        t_base_us = gt_data.t_sys_us[0]

        if not new_imu_path.exists() or regen:
            imu_data = IMUData(self.imu_path, t_base_us=t_base_us)
            imu_data.save_csv(new_imu_path)

        if self.using_cam and using_cam and (not new_cam_path.exists() or regen):
            cam_data = ARCoreData(self.cam_path, z_up=is_z_up, t_base_us=t_base_us)
            cam_data.save_csv(new_cam_path)

        # 复制标定文件
        assert self.calibr_path.exists(), (
            f"Calibration file not found: {self.calibr_path}"
        )
        new_calibr_path = target_unit_path / "Calibration.json"
        if not new_calibr_path.exists():
            self.copy_file(self.calibr_path, new_calibr_path)

        # 复制其他文件
        other_files = ["DataCheck.json", "TimeDiff.png", "Trajectory.png"]
        for file_name in other_files:
            file = self.base_dir / file_name
            self.copy_file(file, target_unit_path / file_name)

        return target_unit_path

    def compress_catch(self, target: Target, *, regen: bool = False):
        try:
            self.compress(target, regen=regen)
        except Exception as e:
            print(f"Error compressing {self.base_dir}: {e}")
            return e


class DeviceData:
    base_dir: Path
    device_name: str
    units: list[CompressUnitData]
    all_units: list[CompressUnitData]
    fail_units: list[CompressUnitData]

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.device_name = self.base_dir.name

        self.all_units = [
            CompressUnitData(it, self.device_name)
            for it in self.base_dir.iterdir()
            if it.is_dir()
        ]
        self.fail_units = [it for it in self.all_units if it.err_msg]
        self.units = [it for it in self.all_units if not it.err_msg]

    def flatten(self):
        return [unit for unit in self.units]


class RuijieDataset:
    base_dir: Path
    devices: list[DeviceData]

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.devices = [DeviceData(it) for it in self.base_dir.iterdir() if it.is_dir()]

    def flatten(self):
        units = []
        for device in self.devices:
            units.extend(device.flatten())
        return units


def move_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = f"mv {src} {dst}"
    os.system(cmd)
    print(f"Moved {src} to {dst}")


if __name__ == "__main__":
    args = DatasetArgsParser()
    args.parser.add_argument(
        "-t", "--type", choices=["navio", "ruijie"], default="navio"
    )
    args.parser.add_argument("-z", "--z_up", action="store_true", help="Z-UP坐标系")
    args.parser.add_argument("--opt", action="store_true", help="使用优化后的数据")
    args.parse()
    assert args.output is not None
    type = args.args.type
    regen = args.regen
    output_path = Path(args.output)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    tg = Target(output_path)

    def action(ud: UnitData):
        ud = CompressUnitData.from_unit(ud)
        ud.compress(tg, regen=regen)

    if args.unit is not None:
        assert args.unit is not None
        ud = UnitData(args.unit)
        action(ud)
    elif args.dataset is not None:
        assert args.dataset is not None
        DatasetDicts = {"ruijie": RuijieDataset, "navio": NavioDataset}
        ds = DatasetDicts[type](args.dataset)
        res = dataset_action(ds, action)

        if len(res):
            print("错误数据：")
            for path, err in res:
                print(f"{path}\n{err}")
