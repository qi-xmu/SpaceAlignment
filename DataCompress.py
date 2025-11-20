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

from base.datatype import ARCoreData, IMUData, RTABData

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


class CompressUnitData:
    data_id: str
    device_name: str
    imu_path: Path
    cam_path: Path
    gt_path: Path
    calibr_path: Path
    err_msg = str | None

    def __init__(self, base_dir: str | Path, device_name: str):
        self.base_dir = Path(base_dir)
        self.data_id = self.base_dir.name
        self.device_name = device_name
        self.err_msg = None

        try:
            self.cam_path = self.base_dir.joinpath("cam.csv")
            self.imu_path = self.base_dir.joinpath("imu.csv")
            self.gt_path = self.base_dir.joinpath("rtab.csv")
            gt_path = self._load_gt_path()
            assert gt_path is not None, f"rtab.csv/*.db not found in {self.base_dir}"
            assert self.cam_path.exists(), f"cam.csv not found in {self.base_dir}"
            assert self.imu_path.exists(), f"imu.csv not found in {self.base_dir}"
            self.gt_path = gt_path
        except Exception as e:
            self.err_msg = e.__str__()
            return

    def _load_gt_path(self):
        # 优先使用 rtab.csv 文件
        gt_path = self.gt_path
        if gt_path.exists():
            return

        # 检查此文件夹下文件，选中第一个后缀为 db 的文件
        for file in self.base_dir.iterdir():
            if file.suffix == ".db":
                gt_path = file
                break
        else:
            raise FileNotFoundError(f"No .db file found in {self.base_dir}")
        return gt_path

    @staticmethod
    def copy_file(src: Path, dst: Path):
        """Copy file from src to dst."""
        dst.write_bytes(src.read_bytes())

    def compress(self, target: Target, *, regen: bool = False):
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

        # if not new_cam_path.exists() or regen:
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
    parser.add_argument("-t", "--type", choices=["ruijie", "navio"], default="navio")
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    regen = args.regen

    if not output_path.exists():
        output_path.mkdir(parents=True)

    rd = RuijieDataset(dataset_path)
    tg = Target(output_path)
    res = []
    for i, device in enumerate(rd.devices):
        for j, unit in enumerate(device.units):
            print(f"\n{i}-{j} {unit.base_dir} ...")
            err = unit.compress_catch(tg, regen=regen)
            if err:
                res.append((unit.base_dir, err))
                print(f"错误信息：{err}")
            else:
                print("成功")

    if len(res):
        print("错误数据：")
        for path, err in res:
            print(path, err)
            move_dir(path, output_path.parent / "Error")
            with open(output_path / "Error" / "error.log", "a") as f:
                print(f"{path} {err}", file=f)
