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

import argparse
import os
from pathlib import Path

from base.action import dataset_action
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
            .joinpath(f"{data_id}_{device_name}")
        )
        imu_path = unit_path.joinpath("imu.csv")
        cam_path = unit_path.joinpath("cam.csv")
        gt_path = unit_path.joinpath("rtab.csv")
        return (unit_path, imu_path, cam_path, gt_path)


class CompressUnitData(UnitData):
    def __init__(self, base_dir: str | Path, device_name: str):
        UnitData.__init__(self, base_dir)
        self.device_name = device_name  # type: ignore

    @staticmethod
    def copy_file(src: Path, dst: Path):
        """Copy file from src to dst."""
        dst.write_bytes(src.read_bytes())

    def compress(self, target: Target, *, regen: bool = False, using_cam: bool = False):
        if self.err_msg:
            return
        t_base_us = 0
        unit_path, new_imu_path, new_cam_path, new_gt_path = target.unit(
            self.data_id, self.device_name
        )

        if not new_imu_path.exists() or regen:
            gt_data = RTABData(self.gt_path, is_load_opt=False)
            unit_path.mkdir(parents=True, exist_ok=True)
            gt_data.save_csv(new_gt_path)
        else:
            gt_data = RTABData(new_gt_path)
        t_base_us = gt_data.t_sys_us[0]

        if not new_imu_path.exists() or regen:
            imu_data = IMUData(self.imu_path, t_base_us=t_base_us)
            imu_data.save_csv(new_imu_path)

        if using_cam and (not new_cam_path.exists() or regen):
            cam_data = ARCoreData(self.cam_path, z_up=True, t_base_us=t_base_us)
            cam_data.save_csv(new_cam_path)

        return unit_path

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
    parser = argparse.ArgumentParser(description="Compress data")
    parser.add_argument("-d", "--dataset", help="Path to the dataset file")
    parser.add_argument("-o", "--output", help="Path to the output file")
    parser.add_argument("-r", "--regen", action="store_true", help="Regenerate data")
    parser.add_argument("-t", "--type", choices=["navio", "ruijie"], default="navio")
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    regen = args.regen

    if not output_path.exists():
        output_path.mkdir(parents=True)

    DatasetDicts = {"ruijie": RuijieDataset, "navio": NavioDataset}
    ds = DatasetDicts[args.type](dataset_path)
    tg = Target(output_path)
    res = []

    def action(ud: CompressUnitData):
        ud.compress(tg, regen=regen)

    dataset_action(ds, action)

    if len(res):
        print("错误数据：")
        for path, err in res:
            print(path, err)
            move_dir(path, output_path.parent / "Error")
            with open(output_path / "Error" / "error.log", "a") as f:
                print(f"{path} {err}", file=f)
