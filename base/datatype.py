import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Literal
import os

SceneType = Literal["in", 'out']  # 场景类型，in 表示室内场景，out 表示室外场景
DeviceType = Literal[
    "SM-G9900",         # 三星 FE 21 5G
    "Redmi K30 Pro",    # 红米 k30 pro
    "ABR-AL60"          # 华为 Mate 60e
]
Pose = tuple[np.ndarray, np.ndarray]


@dataclass
class UnitData:
    data_id: str
    device_type: DeviceType
    gt_path: Path
    imu_path: Path
    cam_path: Path

    def __init__(self, base_dir: Path) -> None:
        ymd, hms, device_type = base_dir.name.split("_")
        self.data_id = f"{ymd}_{hms}"
        self.device_type = device_type  # type: ignore

        self.imu_path = base_dir.joinpath("imu.csv")
        self.cam_path = base_dir.joinpath("cam.csv")

        # 检查此文件夹下文件，选中第一个后缀为 db 的文件
        for file in os.listdir(base_dir):
            if file.endswith(".db"):
                self.gt_path = base_dir.joinpath(file)
                break
        else:
            raise FileNotFoundError(f"No .db file found in {base_dir}")


@dataclass
class GroupData:
    group_id: str
    scene_type: SceneType
    data: list[UnitData]
    calibr_files: dict[str, Path]
    raw_calibr_path: Path

    def __init__(self, base_dir: Path | str):
        base_dir = Path(base_dir)
        ymd, group_id, scene_type = base_dir.name.split("_")
        self.group_id = group_id
        self.scene_type = scene_type  # type: ignore
        self.data = []
        self.calibr_files = {}
        for item in base_dir.iterdir():
            if item.is_dir():
                if item.name.startswith(ymd):
                    self.data.append(UnitData(item))
                elif item.name.startswith("Calibration"):
                    self.raw_calibr_path = item
            if item.is_file() and item.name.endswith(".json"):
                name, suffix = item.name.split(".")
                _, device_type = name.split("_")
                self.calibr_files[device_type] = item

        assert self.raw_calibr_path is not None, f"No raw_calibr_path found in {base_dir}"


@dataclass
class PersonData:
    person_id: str
    groups: list[GroupData]

    def __init__(self, base_dir: Path, person_id: str) -> None:
        self.person_id = person_id
        base_dir = base_dir.joinpath(person_id)
        self.groups = []
        for item in base_dir.iterdir():
            if item.is_dir():
                self.groups.append(GroupData(item))


@dataclass
class FlattenUnitData(UnitData):
    person_id: str | None
    group_id: str
    scene_type: SceneType
    calibr_file: Path

    def __init__(self, person: PersonData | None, group: GroupData, unit: UnitData):
        self.person_id = person.person_id if person is not None else None
        self.group_id = group.group_id
        self.scene_type = group.scene_type
        self.calibr_file = group.calibr_files[unit.device_type]
        self.calibr_data = CalibrationData(self.calibr_file)

        super().__init__(unit.cam_path.parent)

    def parse_calibr_file(self):
        with open(self.calibr_file, "r") as f:
            data = json.load(f)

        R_gc = np.array(data["rotation_matrix"])
        t_gc = np.array(data["translation_vector"])

        return R_gc, t_gc


@dataclass
class CalibrationSeries:
    """
    CalibrationSeries 类用于存储一系列的校准数据，包括时间戳、旋转矩阵和平移向量
    """
    # timestampe in us
    times: list[int] | np.ndarray
    rots: list[np.ndarray]
    trs: list[np.ndarray]

    def get_match(self, matches, index=0, inverse=False):
        Rs = []
        ts = []
        for match in matches:
            rot = self.rots[match[index]]
            tr = self.trs[match[index]]
            if inverse:
                rot = rot.T
                tr = -rot @ tr
            Rs.append(rot)
            ts.append(tr)
        return Rs, ts


@dataclass
class CalibrationData:
    rot_sensor_gt: np.ndarray
    tr_sensor_gt: np.ndarray
    rot_ref_sensor_gt: np.ndarray
    tr_ref_sensor_gt: np.ndarray

    def __init__(self, json_path: Path):
        with open(json_path, "r") as f:
            data = json.load(f)
            self.rot_sensor_gt = np.array(data["rot_sensor_gt"])
            self.tr_sensor_gt = np.array(data["tr_sensor_gt"])
            self.rot_ref_sensor_gt = np.array(data["rot_ref_sensor_gt"])
            self.tr_ref_sensor_gt = np.array(data["tr_ref_sensor_gt"])
