import json
import sqlite3
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyquaternion import Quaternion

from .basetype import DeviceType, Pose, Poses, SceneType, Time  # noqa
from .interpolate import interpolate_vector3d, slerp_quaternion
from .space import OrtR


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
    err_msg: str | None = None
    using_cam: bool

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.data_id = self.base_dir.name
        # device_name
        spl = self.data_id.split("_")
        device_name = spl[2] if len(spl) > 2 else "Unknown"
        self.device_name = device_name  # type: ignore

        self.cam_path = self.base_dir.joinpath("cam.csv")
        self.imu_path = self.base_dir.joinpath("imu.csv")
        self.gt_path = self._load_gt_path()  # self.gt_path

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

        # 使用包含 cam 数据
        try:
            cam_data = np.loadtxt(self.cam_path, delimiter=",")
            self.using_cam = len(cam_data) != 0
        except Exception as _e:
            self.using_cam = False

    def _load_gt_path(self):
        # 优先使用 rtab.csv 文件
        gt_path = self.base_dir.joinpath("rtab.csv")
        if gt_path.exists():
            return gt_path

        # 检查此文件夹下文件，选中第一个后缀为 db 的文件
        for file in self.base_dir.iterdir():
            if file.suffix == ".db":
                gt_path = file
                break
        else:
            self.err_msg = f"No .db file found in {self.base_dir}"
        return gt_path

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
    units: list[UnitData]
    # calibr_files: dict[str, Path]

    def __init__(self, base_dir: Path | str):
        base_dir = Path(base_dir)
        ymd, group_id, scene_type = base_dir.name.split("_")
        self.group_id = group_id
        self.scene_type = scene_type  # type: ignore
        self.units = []

        self.calibr_dir = base_dir.joinpath("Calibration")
        for item in base_dir.iterdir():
            if item.is_dir():
                if item.name.startswith(ymd):
                    self.units.append(UnitData(item))


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


class Dataset:
    root_dir: Path
    persons: list[PersonData]

    def __init__(self, root_dir: str | Path, person_ids: list[str] | None = None):
        self.root_dir = Path(root_dir)
        person_ids = person_ids if person_ids else self._load_dir_list()
        self.persons = [PersonData(self.root_dir.joinpath(pid)) for pid in person_ids]

    def _load_dir_list(self) -> list[str]:
        return [it.name for it in self.root_dir.iterdir() if it.is_dir()]

    def flatten(self) -> list[FlattenUnitData]:
        res = []
        for person in self.persons:
            for group in person.groups:
                for unit in group.units:
                    res.append(FlattenUnitData(person, group, unit))
        return res


class TimePoseSeries:
    """
    CalibrationSeries 类用于存储一系列的校准数据，包括时间戳、旋转矩阵和平移向量
    """

    t_us: Time
    qs: list[Quaternion]
    ps: NDArray[np.float64]

    def __init__(self, t_us: Time, qs: list[Quaternion], ps: NDArray[np.float64]):
        self.t_us = t_us
        self.qs = qs
        self.ps = ps
        self.rate = len(self) / (self.t_us[-1] - self.t_us[0]) * 1e6

    def __len__(self) -> int:
        return len(self.t_us)

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

    def downsample(
        self,
        *,
        factor: int | None = None,
        t_gap: float | None = None,
    ) -> "TimePoseSeries":
        assert factor is not None or t_gap is not None
        if factor is None and t_gap is not None:
            factor = int(np.ceil(t_gap * self.rate))
        assert factor is not None
        factor = max(factor, 1)
        ts = self.t_us[::factor]
        qs = self.qs[::factor]
        ps = self.ps[::factor]
        return TimePoseSeries(ts, qs, ps)

    def interpolate(self, t_new_us: Time):
        qs = slerp_quaternion(
            qs=self.qs,
            t_old_us=self.t_us,
            t_new_us=t_new_us,
        )
        ps = interpolate_vector3d(
            vec3d=self.ps,
            t_old_us=self.t_us,
            t_new_us=t_new_us,
        )
        return TimePoseSeries(t_new_us, qs, ps)

    def get_range(self, start: int | None = None, end: int | None = None):
        start = max(0, int(start * self.rate)) if start is not None else None
        end = min(len(self), int(end * self.rate)) if end is not None else None
        return TimePoseSeries(
            self.t_us[start:end],
            self.qs[start:end],
            self.ps[start:end],
        )


@dataclass
class CalibrationData:
    rot_sensor_gt: NDArray | None
    tr_sensor_gt: NDArray | None
    rot_ref_sensor_gt: NDArray | None
    tr_ref_sensor_gt: NDArray | None

    err_sensor_gt: tuple | None
    err_ref_sensor_gt: tuple | None

    _file_path: Path | None = None

    def __init__(
        self,
        rot_sensor_gt: NDArray | None = None,
        tr_sensor_gt: NDArray | None = None,
        rot_ref_sensor_gt: NDArray | None = None,
        tr_ref_sensor_gt: NDArray | None = None,
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
    def rot_gt_sensor(self) -> NDArray | None:
        if self.rot_sensor_gt is None:
            return None
        return self.rot_sensor_gt.T

    @property
    def tr_gt_sensor(self) -> NDArray | None:
        if self.tr_sensor_gt is None or self.rot_sensor_gt is None:
            return None
        return -self.rot_sensor_gt.T @ self.tr_sensor_gt

    @property
    def rot_ref_gt_sensor(self) -> NDArray | None:
        if self.rot_ref_sensor_gt is None:
            return None
        return self.rot_ref_sensor_gt.T

    @property
    def tr_ref_gt_sensor(self) -> NDArray | None:
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


class IMUColumn:
    """
    #timestamp [us],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],t_system [us]
    #"""

    t = ["#timestamp [us]"]
    w = ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"]
    a = ["a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
    q = ["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]
    t_sys = ["t_system [us]"]

    all = t + w + a + q + t_sys

    pass


class IMUData:
    t_us: Time
    t_us_f0: Time
    t_sys_us: Time
    raw_ahrs: NDArray
    acce: NDArray
    gyro: NDArray
    rate: float

    def __init__(self, file_path: str | Path, *, t_base_us: int = 0) -> None:
        self.file_path = str(file_path)
        self.ahrs_qs: list[Quaternion] = []
        self.load_data(t_base_us)
        self.__len__()

    def __len__(self):
        assert len(self.acce) == len(self.gyro) == len(self.t_sys_us)
        return len(self.t_sys_us)

    def load_data(self, t_base_us: int = 0) -> None:
        """Load IMU data from CSV file."""
        # timestamp [us],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],
        # a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2],
        # q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []
        df: pd.DataFrame = pd.read_csv(self.file_path)
        raw_data = df.to_numpy()

        self.t_us = raw_data[:, 0].astype(np.int64)
        self.gyro = raw_data[:, 1:4]  # angular velocity
        self.acce = raw_data[:, 4:7]  # linear acceleration
        self.raw_ahrs = raw_data[:, 7:11]  # orientation

        # Convert quaternions to unit quaternions
        self.ahrs_qs = [Quaternion(q).unit for q in self.raw_ahrs]

        if len(self.t_us) > 1:
            self.t_us_f0 = self.t_us - self.t_us[0]

        self.extend = bool(raw_data.shape[1] > 11)
        if self.extend:
            self.t_sys_us = raw_data[:, 11].astype(np.int64)  # 1970 us
            self.t_sys_us = self.t_sys_us[0] + self.t_us_f0
        else:
            print("Warning: No system timestamp data available")
            self.t_sys_us = self.t_us_f0 + t_base_us

        # Calculate IMU frequency
        if len(self.t_us) > 1:
            time_diffs = np.diff(self.t_us)
            self.rate = float(1e6 / np.mean(time_diffs))
            print(f"IMU frequency: {self.rate:.2f} Hz")
        else:
            print("Warning: Not enough data points to calculate frequency")

    def get_time_pose_series(self, t_len_s: int | None = None) -> TimePoseSeries:
        max_idx = len(self.t_sys_us)
        if t_len_s is not None:
            max_idx = int(min(t_len_s * self.rate, max_idx))

        return TimePoseSeries(
            t_us=self.t_sys_us[:max_idx],
            qs=self.ahrs_qs[:max_idx],
            ps=np.zeros((len(self.ahrs_qs[:max_idx]), 3)),
        )

    def save_csv(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = np.hstack(
            [
                self.t_us[:, np.newaxis],
                self.gyro,
                self.acce,
                np.array([q.elements for q in self.ahrs_qs]),
                self.t_sys_us[:, np.newaxis],
            ]
        )

        pd.DataFrame(data, columns=IMUColumn.all).to_csv(
            path, index=False, float_format="%.8f"
        )

    def interpolate(self, t_new_us: NDArray):
        self.acce = interpolate_vector3d(
            vec3d=self.acce, t_old_us=self.t_sys_us, t_new_us=t_new_us
        )
        self.gyro = interpolate_vector3d(
            vec3d=self.gyro, t_old_us=self.t_sys_us, t_new_us=t_new_us
        )
        self.ahrs_qs = slerp_quaternion(
            qs=self.ahrs_qs, t_old_us=self.t_sys_us, t_new_us=t_new_us
        )
        self.t_sys_us = t_new_us

    def transform_to_world(
        self, *, rots: NDArray | None = None, qs: list[Quaternion] | None = None
    ):
        if rots is None and qs is not None:
            rots = np.array([q.rotation_matrix for q in qs])
        # 默认使用 AHRS 的数据进行变换
        if rots is None and qs is None:
            rots = np.array([q.rotation_matrix for q in self.ahrs_qs])

        assert rots is not None, "Either rots or qs must be provided"
        assert len(rots) == len(self), (
            f"Length mismatch, got {len(rots)} but expected {len(self)}"
        )
        # rots (i, j, k) acce (i, k) -> (i, j)  (3, 3) (3,1) -> (3,1)
        self.world_acce = np.einsum("ijk,ik->ij", rots, self.acce)  # type: ignore
        self.world_gyro = np.einsum("ijk,ik->ij", rots, self.gyro)  # type: ignore


class ARCoreColumn:
    """
    #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],p_CS_C_x [m],p_CS_C_y [m],p_CS_C_z [m],q_CS_w [],q_CS_x [],q_CS_y [],q_CS_z [],t_system [us]
    """

    t = ["#timestamp [us]"]
    ps = ["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]
    qs = ["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]
    pc = ["p_CS_C_x [m]", "p_CS_C_y [m]", "p_CS_C_z [m]"]
    qc = ["q_CS_w []", "q_CS_x []", "q_CS_y []", "q_CS_z []"]
    t_sys = ["t_system [us]"]

    all = t + ps + qs + pc + qc + t_sys


class ARCoreData:
    rate: float

    def __init__(self, file_path, dataset_id=None, *, z_up=False, t_base_us: int = 0):
        self.file_path = file_path
        self.z_up = z_up

        if self.z_up:
            self.base_sensor_cam = Quaternion()
        else:
            self.base_sensor_cam = Quaternion(axis=[1, 0, 0], angle=np.pi / 2)

        self.load_data(t_base_us)

    def __len__(self):
        return self.sensor_t_us.__len__()

    def _transform_world(self, ps):
        return np.einsum("ij,kj->ki", self.base_sensor_cam.rotation_matrix, ps)

    def _to_q_obj(self, qs):
        return [(self.base_sensor_cam * Quaternion(q)).unit for q in qs]

    def load_data(self, t_base_us=0):
        # #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []
        self.raw_data = pd.read_csv(self.file_path).to_numpy()

        self.sensor_t_us = self.raw_data[:, 0]
        self.raw_sensor_ps = self.raw_data[:, 1:4]
        self.raw_sensor_qs = self.raw_data[:, 4:8]  # wxyz

        # convert to Quaternion object
        self.t_us_f0 = self.sensor_t_us - self.sensor_t_us[0]
        self.sensor_qs = self._to_q_obj(self.raw_sensor_qs)
        self.sensor_ps = self._transform_world(self.raw_sensor_ps)
        self.rate = float(1e6 / np.mean(np.diff(self.sensor_t_us)))

        self.extend = self.raw_data.shape[1] > 8
        if self.extend:
            self.raw_cam_ps = self.raw_data[:, 8:11]
            self.raw_cam_qs = self.raw_data[:, 11:15]
            self.t_sys_us = self.raw_data[:, 15]

            self.cam_ps = self._transform_world(self.raw_cam_ps)
            self.cam_qs = self._to_q_obj(self.raw_cam_qs)
            # 根据 self.t_us 的时间间隔更新 self.t_sys_us，使其与 self.t_us 保持一致
            # 使用 t_us 的时间基准，但保持 t_sys_us 的起始时间
            self.t_sys_us = self.t_sys_us[0] + self.t_us_f0
        else:
            print("Warning: No extended data available")
            self.cam_ps = self.sensor_ps
            self.cam_qs = self.sensor_qs
            self.t_sys_us = self.t_us_f0 + t_base_us

    def get_time_pose_series(
        self, t_len_s: int | None = None, *, using_cam: bool = False
    ) -> TimePoseSeries:
        max_idx = len(self.t_sys_us)
        if t_len_s is not None:
            max_idx = min(max_idx, int(t_len_s * 1e6))
        return TimePoseSeries(
            t_us=self.t_sys_us[:max_idx],
            qs=self.sensor_qs[:max_idx] if not using_cam else self.cam_qs[:max_idx],
            ps=self.sensor_ps[:max_idx] if not using_cam else self.cam_ps[:max_idx],
        )

    def save_csv(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = np.hstack(
            [
                self.sensor_t_us[:, np.newaxis],
                self.sensor_ps,
                np.array([q.elements for q in self.sensor_qs]),
                self.cam_ps,
                np.array([q.elements for q in self.cam_qs]),
                self.t_sys_us[:, np.newaxis],
            ]
        )

        pd.DataFrame(data, columns=ARCoreColumn.all).to_csv(
            path, index=False, float_format="%.8f"
        )

    def draw(self, show=True):
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(
                self.raw_sensor_ps[:, 0],
                self.raw_sensor_ps[:, 1],
                label="ARCore Trajectory",
            )
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.grid(True)
            ax.set_aspect("equal", "box")
            ax.legend()
            if show:
                fig.show()
        except Exception as e:
            print(f"Error in drawing ARCore data: {e}")


class RTABData:
    rate: float
    t_len_s: float

    def __init__(
        self,
        file_path: str | Path,
        calibr_data: CalibrationData | None = None,
        *,
        is_load_opt: bool = True,
    ):
        self.file_path = str(file_path)
        self.calibr_data = calibr_data

        self.opt_ids: list[int] = []
        self.opt_qs: list[Quaternion] = []
        self.opt_ps: NDArray
        self.opt_t_us: NDArray[np.int64]

        self.node_t_us: NDArray[np.int64]
        self.t_us_f0: NDArray[np.int64]
        self.node_ids: list[int] = []
        self.node_qs: list[Quaternion] = []
        self.node_ps: NDArray

        if self.file_path.endswith(".csv"):
            print("Loading from CSV data...")
            self.load_csv_data()
        elif self.file_path.endswith(".db"):
            print("Loading from DB data...")
            # 从数据库中加载
            self.conn = sqlite3.connect(self.file_path)
            self.cursor = self.conn.cursor()
            self.load_node_data()
            if is_load_opt:
                self.load_opt_data()
        else:
            raise ValueError("RTAB-Map data file must be a .csv or .db file")

        self.t_us_f0 = self.node_t_us - self.node_t_us[0]
        self.t_sys_us = self.node_t_us
        self.rate = float(1e6 / np.mean(np.diff(self.node_t_us)))
        self.t_len_s = (self.node_t_us[-1] - self.node_t_us[0]) / 1e6
        print(
            f"Loaded {len(self.node_ids)} nodes, {len(self.opt_ids)} optimized witch Freq: {self.rate:.2f} from the database. "
        )

    def load_csv_data(self):
        # #timestamp [us],p_RN_x [m],p_RN_y [m],p_RN_z [m],q_RN_w [],q_RN_x [],q_RN_y [],q_RN_z []
        data = pd.read_csv(self.file_path).to_numpy()
        self.node_t_us = data[:, 0]
        self.node_ps = data[:, 1:4]
        self.node_qs = [Quaternion(*q).unit for q in data[:, 4:]]
        self.opt_qs = self.node_qs
        self.opt_ps = self.node_ps

    @classmethod
    def _decompress_data(cls, blob_data) -> bytes | None:
        """通用的解压缩和解析函数"""
        try:
            # 尝试解压缩数据
            try:
                decompressed_data = zlib.decompress(blob_data)
            except Exception:
                decompressed_data = blob_data

            return decompressed_data
        except Exception as e:
            print(f"Error decompressing data: {e}")
            return None

    @classmethod
    def _unpack_pose_data(cls, blob_data):
        """解压并解析RTAB-Map的pose数据"""
        decompressed_data = cls._decompress_data(blob_data)
        if decompressed_data and len(decompressed_data) >= 48:
            pose_values = struct.unpack("12f", decompressed_data[:48])
            pose_matrix = np.array(pose_values).reshape(3, 4)
            p = pose_matrix[:3, 3]
            R = pose_matrix[:3, :3]
            U, _, Vt = np.linalg.svd(R)
            R_orthogonal = U @ Vt @ np.diag([1, 1, np.linalg.det(U @ Vt)])
            unit_q = Quaternion(matrix=R_orthogonal).unit
            # unit_q = rotation_matrix_to_quaternion(R).unit
            return p, unit_q
        return None

    def load_opt_data(self):
        # 查询Admin表中的opt_poses数据
        admin_opt_poses = self.cursor.execute(
            "SELECT opt_poses FROM Admin WHERE opt_poses IS NOT NULL"
        ).fetchone()
        assert admin_opt_poses is not None, "Failed to fetch admin_opt_poses data"
        decompressed_data = self._decompress_data(admin_opt_poses[0])

        assert decompressed_data is not None, (
            "Failed to decompress admin_opt_poses data"
        )

        opt_ps = []
        for pose_values in struct.iter_unpack("12f", decompressed_data):
            pose_matrix = np.array(pose_values).reshape(3, 4)

            R = pose_matrix[:3, :3]
            p = pose_matrix[:3, 3]
            unit_q = Quaternion(matrix=OrtR(R)).unit
            opt_ps.append(p)
            self.opt_qs.append(unit_q)

            # calibr
            if self.calibr_data:
                pass
        self.opt_ps = np.array(opt_ps)
        # 查询Admin表中的 opt_ids 数据
        admin_opt_ids = self.cursor.execute(
            "SELECT opt_ids FROM Admin WHERE opt_ids IS NOT NULL"
        ).fetchone()

        decompressed_ids = self._decompress_data(admin_opt_ids[0])
        assert decompressed_ids is not None, "Failed to decompress admin_opt_ids data"

        self.opt_ids = list(
            struct.unpack(f"{len(decompressed_ids) // 4}i", decompressed_ids)
        )
        assert (
            self.opt_ids.__len__() == self.opt_ps.__len__() == self.opt_qs.__len__()
        ), "Mismatch in lengths of opt_ids, ps, and unit_qs"

        # 通过 ids 获取时间戳
        opt_t_us = [self.node_t_us[self.node_ids.index(idx)] for idx in self.opt_ids]
        self.opt_t_us = np.array(opt_t_us)

    def load_node_data(self):
        results = self.cursor.execute("""
            SELECT id, stamp, pose
            FROM Node
            WHERE pose IS NOT NULL
            ORDER BY id
        """).fetchall()

        assert results, f"No node data found in the database. {self.file_path}"

        node_t_us = []
        node_ps = []
        for node_id, stamp, pose_blob in results:
            pose = self._unpack_pose_data(pose_blob)
            if pose is None:
                continue

            p, unit_q = pose
            self.node_ids.append(node_id)
            node_t_us.append(int(stamp * 1e6))  # convert to us
            node_ps.append(p)
            self.node_qs.append(unit_q)

        self.node_t_us = np.array(node_t_us)
        self.node_ps = np.array(node_ps)

    def get_time_pose_series(
        self, t_len_s: int | None = None, *, using_opt: bool = False
    ) -> TimePoseSeries:
        max_idx = len(self.node_t_us)
        if t_len_s is not None:
            max_idx = min(max_idx, int(t_len_s * self.rate))
        return TimePoseSeries(
            # self.node_t_us  == self.t_sys_us
            t_us=self.t_sys_us[:max_idx] if not using_opt else self.opt_t_us[:max_idx],
            qs=self.node_qs[:max_idx] if not using_opt else self.opt_qs[:max_idx],
            ps=self.node_ps[:max_idx] if not using_opt else self.opt_ps[:max_idx],
        )

    def fix_time(self, t_21_us: int):
        self.node_t_us += t_21_us
        self.t_us_f0 += t_21_us
        self.opt_t_us += t_21_us

    def interpolate(self, t_new_us: NDArray):
        self.node_qs = slerp_quaternion(
            qs=self.node_qs, t_old_us=self.t_sys_us, t_new_us=t_new_us
        )
        self.node_ps = interpolate_vector3d(
            vec3d=self.node_ps, t_old_us=self.t_sys_us, t_new_us=t_new_us
        )
        self.t_sys_us = t_new_us

    def draw(
        self,
        *,
        mark_idxs: tuple[list[int], list | None] = ([], None),
        show=True,
        save_path=None,
    ):
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # 绘制开始点和结束点
            ps = np.array(self.opt_ps)
            ax.plot(ps[0, 0], ps[0, 1], "o", color="g", label="Start")
            ax.plot(ps[-1, 0], ps[-1, 1], "x", color="r", label="End")
            ax.plot(ps[:, 0], ps[:, 1], label="Optimized Trajectory", color="b")

            node_ps = np.array(self.node_ps)
            ax.plot(node_ps[0, 0], node_ps[0, 1], "o", color="g")
            ax.plot(node_ps[-1, 0], node_ps[-1, 1], "x", color="r")
            ax.plot(
                node_ps[:, 0],
                node_ps[:, 1],
                label="Node Trajectory",
                color="r",
                alpha=0.5,
            )

            # 绘制标记点
            if mark_idxs:
                mark_note = (
                    [str(it) for it in mark_idxs[1]]
                    if mark_idxs[1] is not None
                    else ["" for _ in range(len(mark_idxs[0]))]
                )
                mark_ps = node_ps[mark_idxs[0]]
                assert len(mark_ps) == len(mark_note), (
                    f"mark_idxs and mark_note must have the same length, but got {len(mark_idxs)} and {len(mark_note)}"
                )
                for i in range(len(mark_ps)):
                    ax.plot(mark_ps[i, 0], mark_ps[i, 1], "o", color="y")
                    ax.text(mark_ps[i, 0], mark_ps[i, 1], mark_note[i], color="y")

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"{self.file_path} Trajectory")
            ax.grid(True)
            ax.set_aspect("equal", "box")
            # 设置 字体大小
            ax.legend(fontsize=8)
            if show:
                fig.show()

            if save_path:
                fig.savefig(save_path, dpi=300)
        except ImportError:
            print("matplotlib is required for drawing the trajectory.")

    def save_csv(self, path: str | Path):
        # #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],
        header = [
            "#timestamp [us]",
            "p_RN_x [m]",
            "p_RN_y [m]",
            "p_RN_z [m]",
            "q_RN_w []",
            "q_RN_x []",
            "q_RN_y []",
            "q_RN_z []",
            # "p_RO_x [m]",
            # "p_RO_y [m]",
            # "p_RO_z [m]",
            # "q_RO_w []",
            # "q_RO_x []",
            # "q_RO_y []",
            # "q_RO_z []",
        ]

        data = np.concatenate(
            [
                self.t_sys_us.reshape(-1, 1),
                self.node_ps,
                np.array([q.elements for q in self.node_qs]),
                # self.opt_ps,
                # np.array([q.elements for q in self.opt_qs])
            ],
            axis=1,
        )

        pd.DataFrame(data, columns=header).to_csv(
            path, index=False, float_format="%.8f"
        )
        print(f"Saved {len(self.node_ids)} poses to {path}")
