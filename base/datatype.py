import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pyquaternion import Quaternion

Pose = tuple[np.ndarray | None, np.ndarray | None]
Poses = tuple[list[np.ndarray], list[np.ndarray]]
Time = NDArray[np.int64]

NO_CAM_DEVICES = ["ABR-AL60"]
SceneType = Literal["in", "out"]  # 场景类型，in 表示室内场景，out 表示室外场景
DeviceType = Literal[
    "SM-G9900",  # 三星 FE 21 5G
    "Redmi K30 Pro",  # 红米 k30 pro
    "ABR-AL60",  # 华为 Mate 60e,
    "Unknown",  # 未知设备
]


class UnitData:
    _CALIBR_FILE = "Calibration_{}.json"

    data_id: str
    device_name: DeviceType
    imu_path: Path
    cam_path: Path  # ARCore
    gt_path: Path
    calibr_path: Path
    is_z_up: bool
    is_calibr_data: bool

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.data_id = self.base_dir.name
        # device_name
        spl = self.data_id.split("_")
        device_name = spl[2] if len(spl) > 2 else "Unknown"
        self.device_name = device_name  # type: ignore

        self.cam_path = self.base_dir.joinpath("cam.csv")
        self.imu_path = self.base_dir.joinpath("imu.csv")
        self._load_gt_path()  # self.gt_path

        # 获取 组 名称
        self.group_path = self.base_dir.parent
        self.is_calibr_data = "/Calibration" in str(self.base_dir)
        if self.is_calibr_data:
            self.group_path = self.group_path.parent
        # 标定文件
        calibr_file = self.group_path.joinpath(self._CALIBR_FILE.format(device_name))
        if not calibr_file.exists():
            calibr_file = self.base_dir.joinpath("Calibration.json")
        self.calibr_path = calibr_file
        self.is_z_up = False

    @property
    def using_cam(self):
        return self.device_name not in NO_CAM_DEVICES

    def _load_gt_path(self):
        # 优先使用 rtab.csv 文件
        self.gt_path = self.base_dir.joinpath("rtab.csv")
        if self.gt_path.exists():
            return

        # 检查此文件夹下文件，选中第一个后缀为 db 的文件
        for file in self.base_dir.iterdir():
            if file.suffix == ".db":
                self.gt_path = file
                break
        else:
            raise FileNotFoundError(f"No .db file found in {self.base_dir}")

    def target(self, target_name) -> Path:
        return self.base_dir.joinpath(target_name)

    def __str__(self) -> str:
        return str(
            {
                "imu_path": self.imu_path,
                "cam_path": self.cam_path,
                "gt_path": self.gt_path,
                "calir_file": self.calibr_path,
            }
        )


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

        assert self.raw_calibr_path is not None, (
            f"No raw_calibr_path found in {base_dir}"
        )


@dataclass
class PersonData:
    person_id: str
    groups: list[GroupData]

    def __init__(self, base_dir: Path) -> None:
        self.person_id = base_dir.name
        self.groups = []
        for item in base_dir.iterdir():
            if item.is_dir():
                self.groups.append(GroupData(item))


@dataclass
class FlattenUnitData(UnitData):
    person_id: str | None
    group_id: str
    scene_type: SceneType

    def __init__(self, person: PersonData | None, group: GroupData, unit: UnitData):
        self.person_id = person.person_id if person is not None else None
        self.group_id = group.group_id
        self.scene_type = group.scene_type
        super().__init__(unit.base_dir)


class TimePoseSeries:
    """
    CalibrationSeries 类用于存储一系列的校准数据，包括时间戳、旋转矩阵和平移向量
    """

    t_us: Time
    qs: list[Quaternion]
    ps: NDArray

    def __init__(self, t_us: Time, qs: list[Quaternion], ps: NDArray):
        self.t_us = t_us
        self.qs = qs
        self.ps = ps

    def __len__(self) -> int:
        return len(self.t_us)

    @property
    def rate(self) -> float:
        return len(self) / (self.t_us[-1] - self.t_us[0]) * 1e6

    def get_match(self, matches, *, index=0, inverse=False) -> Poses:
        Rs = []
        ts = []
        for match in matches:
            rot = self.qs[match[index]].rotation_matrix
            tr = self.ps[match[index]]
            if inverse:
                rot = rot.T
                tr = -rot @ tr
            Rs.append(rot)
            ts.append(tr)
        return Rs, ts

    def get_all(self, *, inverse=False) -> Poses:
        Rs = []
        ts = []
        for i in range(len(self)):
            rot = self.qs[i].rotation_matrix
            tr = self.ps[i]
            if inverse:
                rot = rot.T
                tr = -rot @ tr
            Rs.append(rot)
            ts.append(tr)
        return Rs, ts

    def match_series(self, matches, index=0) -> "TimePoseSeries":
        ts = []
        qs = []
        ps = []
        for match in matches:
            ts.append(self.t_us[match[index]])
            qs.append(self.qs[match[index]])
            ps.append(self.ps[match[index]])
        ts = np.array(ts)
        ps = np.array(ps)
        return TimePoseSeries(ts, qs, ps)

    def get_range(self, start: int | None = None, end: int | None = None):
        return TimePoseSeries(
            self.t_us[start:end],
            self.qs[start:end],
            self.ps[start:end],
        )


@dataclass
class CalibrationData:
    rot_sensor_gt: np.ndarray | None
    tr_sensor_gt: np.ndarray | None
    rot_ref_sensor_gt: np.ndarray | None
    tr_ref_sensor_gt: np.ndarray | None

    err_sensor_gt: tuple | None
    err_ref_sensor_gt: tuple | None

    _file_path: Path | None = None

    def __init__(
        self,
        rot_sensor_gt: np.ndarray | None = None,
        tr_sensor_gt: np.ndarray | None = None,
        rot_ref_sensor_gt: np.ndarray | None = None,
        tr_ref_sensor_gt: np.ndarray | None = None,
        err_sensor_gt: tuple | None = None,
        err_ref_sensor_gt: tuple | None = None,
        file_path: Path | None = None,
    ):
        self.rot_sensor_gt = rot_sensor_gt
        self.tr_sensor_gt = tr_sensor_gt
        self.rot_ref_sensor_gt = rot_ref_sensor_gt
        self.tr_ref_sensor_gt = tr_ref_sensor_gt
        self.err_sensor_gt = err_sensor_gt
        self.err_ref_sensor_gt = err_ref_sensor_gt
        self._file_path = file_path

    @property
    def rot_gt_sensor(self) -> np.ndarray | None:
        if self.rot_sensor_gt is None:
            return None
        return self.rot_sensor_gt.T

    @property
    def tr_gt_sensor(self) -> np.ndarray | None:
        if self.tr_sensor_gt is None or self.rot_sensor_gt is None:
            return None
        return -self.rot_sensor_gt.T @ self.tr_sensor_gt

    @property
    def rot_ref_gt_sensor(self) -> np.ndarray | None:
        if self.rot_ref_sensor_gt is None:
            return None
        return self.rot_ref_sensor_gt.T

    @property
    def tr_ref_gt_sensor(self) -> np.ndarray | None:
        if self.tr_ref_sensor_gt is None or self.rot_ref_sensor_gt is None:
            return None
        return -self.rot_ref_sensor_gt.T @ self.tr_ref_sensor_gt

    @property
    def tf_local(self):
        return (self.rot_sensor_gt, self.tr_sensor_gt)

    @property
    def tf_world(self):
        return (self.rot_ref_gt_sensor, self.tr_ref_gt_sensor)

    @staticmethod
    def from_json(json_path: Path) -> "CalibrationData":
        with open(json_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError("Invalid JSON format")
            data = data[0]

            return CalibrationData(
                rot_sensor_gt=np.array(data["rot_sensor_gt"]),
                tr_sensor_gt=np.array(data["trans_sensor_gt"]).flatten(),
                rot_ref_sensor_gt=np.array(data["rot_ref_sensor_gt"]),
                tr_ref_sensor_gt=np.array(data["trans_ref_sensor_gt"]).flatten(),
                file_path=json_path,
            )

    def to_json(self, json_path: Path | str, notes_ext: str = "") -> None:
        rot_sensor_gt = (
            self.rot_sensor_gt.tolist() if self.rot_sensor_gt is not None else ""
        )
        rot_ref_sensor_gt = (
            self.rot_ref_sensor_gt.tolist()
            if self.rot_ref_sensor_gt is not None
            else ""
        )
        tr_sensor_gt = (
            self.tr_sensor_gt.tolist() if self.tr_sensor_gt is not None else ""
        )
        tr_ref_sensor_gt = (
            self.tr_ref_sensor_gt.tolist() if self.tr_ref_sensor_gt is not None else ""
        )
        err_sensor_gt = (
            list(self.err_sensor_gt) if self.err_sensor_gt is not None else []
        )
        err_ref_sensor_gt = (
            list(self.err_ref_sensor_gt) if self.err_ref_sensor_gt is not None else []
        )

        notes = (
            "Pose: Sensor_Groundtruth,Error: [mean_rot_err, mean_trans_err, max_rot_err, max_trans_err]. "
            + notes_ext
        )

        with open(json_path, "w") as f:
            json.dump(
                [
                    {
                        "rot_sensor_gt": rot_sensor_gt,
                        "trans_sensor_gt": tr_sensor_gt,
                        "rot_ref_sensor_gt": rot_ref_sensor_gt,
                        "trans_ref_sensor_gt": tr_ref_sensor_gt,
                        "err_sensor_gt": err_sensor_gt,
                        "err_ref_sensor_gt": err_ref_sensor_gt,
                        "file_path": str(json_path),
                        "notes": notes,
                    }
                ],
                f,
                indent=4,
                ensure_ascii=False,
            )
            print(f"标定结果已保存到: {json_path}")
