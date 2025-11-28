import json
import sqlite3
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from typing_extensions import override

from .basetype import DeviceType, PoseSeries, SceneType, Time, Transform  # noqa
from .interpolate import interpolate_vector3d, slerp_rotation


class UnitData:
    _CALIBR_FILE = "Calibration_{}.json"

    data_id: str
    imu_path: Path
    cam_path: Path  # ARCore
    gt_path: Path
    # extend proterty
    device_name: DeviceType
    calibr_path: Path
    is_z_up: bool
    is_calibr_data: bool
    using_cam: bool

    err_msg: str | None = None

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.data_id = self.base_dir.name
        # 设备名称
        spl = self.data_id.split("_")
        device_name = spl[2] if len(spl) > 2 else "Unknown"
        self.device_name = device_name  # type: ignore

        self.cam_path = self.base_dir.joinpath("cam.csv")
        self.imu_path = self.base_dir.joinpath("imu.csv")
        self.gt_path = self._load_gt_path()

        self.check_file = self.base_dir / "DataCheck.json"

        # 获取 组 名称
        self.group_path = self.base_dir.parent
        self.is_calibr_data = "/Calibration" in self.base_dir.as_posix()
        if self.is_calibr_data:
            self.group_path = self.group_path.parent

        # 标定文件
        self.unit_calib_path = self.base_dir.joinpath("Calibration.json")
        self.group_calibr_path = self.group_path.joinpath(
            self._CALIBR_FILE.format(device_name)
        )
        self.calibr_path = self.unit_calib_path
        self.is_z_up = True

        # 使用包含 cam 数据
        try:
            cam_data = np.loadtxt(self.cam_path, delimiter=",")
            self.using_cam = len(cam_data) != 0
        except Exception as _e:
            self.using_cam = False

    def _load_gt_path(self):
        # 优先使用 gt.csv 文件
        gt_path = self.base_dir / "gt.csv"
        if gt_path.exists():
            return gt_path

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


class Dataset:
    def flatten(self) -> list[UnitData]:
        raise NotImplementedError


@dataclass
class GroupData(Dataset):
    group_id: str
    scene_type: SceneType
    units: list[UnitData]
    calib_units: list[UnitData]
    # calibr_files: dict[str, Path]

    def __init__(self, base_dir: Path | str):
        base_dir = Path(base_dir)
        ymd, group_id, scene_type = base_dir.name.split("_")
        self.group_id = group_id
        self.scene_type = scene_type  # type: ignore
        self.units = []

        self.calibr_dir = base_dir.joinpath("Calibration")
        if self.calibr_dir.exists():
            self.calib_units = [
                UnitData(path) for path in self.calibr_dir.iterdir() if path.is_dir()
            ]
        self.units = [UnitData(path) for path in base_dir.iterdir() if path.is_dir()]

    def flatten(self) -> list[UnitData]:
        return self.units


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

    def flatten(self) -> list[UnitData]:
        units = []
        for group in self.groups:
            units.extend(group.flatten())
        return units


class NavioDataset(Dataset):
    root_dir: Path
    persons: list[PersonData]

    def __init__(self, root_dir: str | Path, person_ids: list[str] | None = None):
        self.root_dir = Path(root_dir)
        person_ids = (
            person_ids
            if person_ids
            else [it.name for it in self.root_dir.iterdir() if it.is_dir()]
        )
        self.persons = [PersonData(self.root_dir.joinpath(pid)) for pid in person_ids]

    @override
    def flatten(self) -> list[UnitData]:
        units = []
        for person in self.persons:
            units.extend(person.flatten())
        return units


class TimePoseSeries:
    """
    CalibrationSeries 类用于存储一系列的校准数据，包括时间戳、旋转矩阵和平移向量
    """

    t_us: Time
    rots: Rotation
    trans: NDArray

    def __init__(self, t_us: Time, rots: Rotation, ps: NDArray):
        self.t_us = t_us
        self.rots = rots
        self.trans = ps
        self.rate = float(np.mean(1e6 / np.diff(t_us)))

    def __len__(self) -> int:
        return len(self.t_us)

    def get_all(self, *, inverse=False) -> PoseSeries:
        poses = PoseSeries(self.rots, self.trans)
        if inverse:
            poses = poses.inverse()
        return poses

    def interpolate(self, t_new_us: Time):
        rots = slerp_rotation(self.rots, t_old_us=self.t_us, t_new_us=t_new_us)
        ps = interpolate_vector3d(
            vec3d=self.trans, t_old_us=self.t_us, t_new_us=t_new_us
        )
        return TimePoseSeries(t_new_us, rots, ps)

    def transform_global(self, tf: Transform):
        self.rots = tf.rot * self.rots
        self.trans = tf.tran + tf.rot.apply(self.trans)

    def transform_local(self, tf: Transform):
        self.trans = self.trans + self.rots.apply(tf.tran)
        self.rots = self.rots * tf.rot


@dataclass
class CalibrationData:
    tf_local: Transform
    tf_global: Transform
    notes: str = ""

    @classmethod
    def identity(cls):
        return cls(Transform.identity(), Transform.identity())

    @classmethod
    def from_json(cls, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError("Invalid JSON format")
            data = data[0]

            rot_local = np.array(data["rot_sensor_gt"])
            trans_local = np.array(data["trans_sensor_gt"]).flatten()
            tf_sg_local = Transform(Rotation.from_matrix(rot_local), trans_local)

            rot_global = np.array(data["rot_ref_sensor_gt"])
            trans_global = np.array(data["trans_ref_sensor_gt"]).flatten()
            tf_sg_global = Transform(Rotation.from_matrix(rot_global), trans_global)
            return cls(tf_sg_local, tf_sg_global)

    def to_json(self, json_path: Path, notes_ext: str = ""):
        rot_sensor_gt = self.tf_local.rot.as_matrix().tolist()
        tr_sensor_gt = self.tf_local.tran.tolist()
        rot_ref_sensor_gt = self.tf_global.rot.as_matrix().tolist()
        tr_ref_sensor_gt = self.tf_global.tran.tolist()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "rot_sensor_gt": rot_sensor_gt,
                        "trans_sensor_gt": tr_sensor_gt,
                        "rot_ref_sensor_gt": rot_ref_sensor_gt,
                        "trans_ref_sensor_gt": tr_ref_sensor_gt,
                        "file_path": str(json_path),
                        "notes": self.notes + notes_ext,
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
    """

    t = ["#timestamp [us]"]
    w = ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"]
    a = ["a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
    q = ["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]
    t_sys = ["t_system [us]"]
    m = ["m_RS_S_x [µT]", "m_RS_S_y [µT]", "m_RS_S_z [µT]"]
    all = t + w + a + q + t_sys + m


class IMUData:
    t_us: Time
    t_us_f0: Time
    t_sys_us: Time
    raw_ahrs: NDArray
    gyro: NDArray
    acce: NDArray
    magn: NDArray
    rate: float
    ahrs_rots: Rotation

    def __init__(self, file_path: str | Path, *, t_base_us: int = 0) -> None:
        self.file_path = str(file_path)
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
        # mag_x [uT],mag_y [uT],mag_z [uT]
        df: pd.DataFrame = pd.read_csv(self.file_path).dropna()
        raw_data = df.to_numpy()

        self.t_us = raw_data[:, 0]
        self.gyro = raw_data[:, 1:4]  # angular velocity
        self.acce = raw_data[:, 4:7]  # linear acceleration
        self.raw_ahrs = raw_data[:, 7:11]  # orientation wxyz
        self.ahrs_rots = Rotation.from_quat(self.raw_ahrs, scalar_first=True)

        assert len(self.t_us) > 1
        self.t_us_f0 = self.t_us - self.t_us[0]
        self.t_sys_us = raw_data[0, 11] + self.t_us_f0
        self.magn = raw_data[:, 12:15]

        # Calculate IMU frequency
        self.rate = float(1e6 / np.mean(np.diff(self.t_us)))
        print(f"IMU frequency: {self.rate:.2f} Hz")

    def get_time_pose_series(self, time_range: tuple = (None, None)) -> TimePoseSeries:
        s_idx, e_idx = None, None
        t_s_s, t_e_s = time_range
        if t_s_s is not None:
            s_idx = int(max(0, t_s_s * self.rate))

        if t_e_s is not None:
            e_idx = int(min(len(self), t_e_s * self.rate))

        return TimePoseSeries(
            t_us=self.t_sys_us[s_idx:e_idx],
            rots=self.ahrs_rots[s_idx:e_idx],
            ps=np.zeros((len(self.t_sys_us[s_idx:e_idx]), 3)),
        )

    def save_csv(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = np.hstack(
            [
                self.t_us.reshape(-1, 1),
                self.gyro,
                self.acce,
                self.ahrs_rots.as_quat(scalar_first=True),
                self.t_sys_us.reshape(-1, 1),
                self.magn,
            ]
        )

        pd.DataFrame(data, columns=IMUColumn.all).to_csv(
            path, index=False, float_format="%.8f"
        )

    def interpolate(self, t_new_us: NDArray):
        self.gyro = interpolate_vector3d(
            vec3d=self.gyro, t_old_us=self.t_sys_us, t_new_us=t_new_us
        )
        self.acce = interpolate_vector3d(
            vec3d=self.acce, t_old_us=self.t_sys_us, t_new_us=t_new_us
        )
        self.ahrs_rots = slerp_rotation(self.ahrs_rots, self.t_sys_us, t_new_us)
        self.t_sys_us = t_new_us

    def transform_to_world(self, *, rots: Rotation | None = None):
        # 默认使用 AHRS 的数据进行变换
        if rots is None:
            rots = self.ahrs_rots

        assert len(rots) == len(self), f"Length mismatch, {len(rots)} != {len(self)}"
        self.world_gyro = rots.apply(self.gyro)
        self.world_acce = rots.apply(self.acce)


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

    def __init__(self, file_path, dataset_id=None, *, z_up=True, t_base_us: int = 0):
        self.file_path = file_path
        self.z_up = z_up

        if self.z_up:
            self.base_sensor_cam = Rotation.identity()
        else:
            self.base_sensor_cam = Rotation.from_euler(
                angles=[np.pi / 2, 0, 0], seq="xyz"
            )

        self.load_data(t_base_us)

    def __len__(self):
        return self.sensor_t_us.__len__()

    def _transform_world(self, ps):
        return self.base_sensor_cam.apply(ps)

    def load_data(self, t_base_us=0):
        # #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []
        self.raw_data = pd.read_csv(self.file_path).to_numpy()

        # Remove duplicate timestamps and ensure strictly increasing
        if len(self.raw_data) > 1:
            # Get unique timestamps while preserving order
            _, unique_indices = np.unique(self.raw_data[:, 0], return_index=True)
            self.raw_data = self.raw_data[unique_indices]

        self.sensor_t_us = self.raw_data[:, 0]
        self.raw_sensor_ps = self.raw_data[:, 1:4]
        self.raw_sensor_qs = self.raw_data[:, 4:8]  # wxyz

        # convert to Quaternion object
        self.t_us_f0 = self.sensor_t_us - self.sensor_t_us[0]
        self.sensor_rots = self.base_sensor_cam * Rotation.from_quat(
            self.raw_sensor_qs, scalar_first=True
        )
        self.sensor_ps = self._transform_world(self.raw_sensor_ps)
        self.rate = float(1e6 / np.mean(np.diff(self.sensor_t_us)))

        self.extend = self.raw_data.shape[1] > 8
        if self.extend:
            self.raw_cam_ps = self.raw_data[:, 8:11]
            self.raw_cam_qs = self.raw_data[:, 11:15]
            self.t_sys_us = self.raw_data[:, 15]

            self.cam_ps = self._transform_world(self.raw_cam_ps)
            self.cam_rots = Rotation.from_quat(self.raw_cam_qs, scalar_first=True)
            # 根据 self.t_us 的时间间隔更新 self.t_sys_us，使其与 self.t_us 保持一致
            # 使用 t_us 的时间基准，但保持 t_sys_us 的起始时间
            self.t_sys_us = self.t_sys_us[0] + self.t_us_f0
        else:
            print("Warning: No extended data available")
            self.cam_ps = self.sensor_ps
            self.cam_rots = self.sensor_rots
            self.t_sys_us = self.t_us_f0 + t_base_us

    def get_time_pose_series(self, time_range: tuple = (None, None)) -> TimePoseSeries:
        s_idx, e_idx = None, None
        t_s_s, t_e_s = time_range
        if t_s_s is not None:
            s_idx = int(max(0, t_s_s * self.rate))
        if t_e_s is not None:
            e_idx = int(min(len(self.t_sys_us), t_e_s * self.rate))
        return TimePoseSeries(
            t_us=self.t_sys_us[s_idx:e_idx],
            rots=self.sensor_rots[s_idx:e_idx],
            ps=self.sensor_ps[s_idx:e_idx],
        )

    def save_csv(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = np.hstack(
            [
                self.sensor_t_us[:, np.newaxis],
                self.sensor_ps,
                self.sensor_rots.as_quat(scalar_first=True),
                self.cam_ps,
                self.cam_rots.as_quat(scalar_first=True),
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


class PosesData:
    pass


class RTABData:
    t_len_s: float
    rate: float
    node_rots: Rotation
    opt_rots: Rotation

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
        self.opt_ps: NDArray
        self.opt_t_us: NDArray[np.int64]

        self.node_t_us: NDArray[np.int64]
        self.t_us_f0: NDArray[np.int64]
        self.node_ids: list[int] = []
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

    def __len__(self):
        return len(self.node_t_us)

    def load_csv_data(self):
        # #timestamp [us],p_RN_x [m],p_RN_y [m],p_RN_z [m],q_RN_w [],q_RN_x [],q_RN_y [],q_RN_z []
        data = pd.read_csv(self.file_path).to_numpy()
        self.node_t_us = data[:, 0]
        self.node_ps = data[:, 1:4]
        self.node_rots = Rotation.from_quat(data[:, 4:], scalar_first=True)
        self.opt_rots = self.node_rots
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
    def _unpack_pose_data(cls, blob_data) -> Transform | None:
        """解压并解析RTAB-Map的pose数据"""
        decompressed_data = cls._decompress_data(blob_data)
        if decompressed_data and len(decompressed_data) >= 48:
            pose_values = struct.unpack("12f", decompressed_data[:48])
            pose_matrix = np.array(pose_values).reshape(3, 4)
            p = pose_matrix[:3, 3]
            R = pose_matrix[:3, :3]
            return Transform.from_raw(R, p)
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

        opt_rots = []
        opt_ps = []
        for pose_values in struct.iter_unpack("12f", decompressed_data):
            pose_matrix = np.array(pose_values).reshape(3, 4)
            R = pose_matrix[:3, :3]
            p = pose_matrix[:3, 3]
            opt_rots.append(R)
            opt_ps.append(p)

            # calibr
            if self.calibr_data:
                pass
        self.opt_rots = Rotation.from_matrix(opt_rots)
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
            self.opt_ids.__len__() == self.opt_ps.__len__() == self.opt_rots.__len__()
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
        node_rots = []
        node_ps = []
        for node_id, stamp, pose_blob in results:
            pose = self._unpack_pose_data(pose_blob)
            if pose is None:
                continue

            self.node_ids.append(node_id)
            node_t_us.append(int(stamp * 1e6))  # convert to us
            node_rots.append(pose.rot.as_quat())
            node_ps.append(pose.tran)

        self.node_t_us = np.array(node_t_us)
        self.node_rots = Rotation.from_quat(node_rots)
        self.node_ps = np.array(node_ps)

    def get_time_pose_series(
        self, time_range: tuple = (None, None), *, using_opt: bool = False
    ) -> TimePoseSeries:
        s_idx, e_idx = None, None
        t_s_s, t_e_s = time_range
        if t_s_s is not None:
            s_idx = int(max(0, t_s_s * self.rate))

        if t_e_s is not None:
            e_idx = int(min(len(self), t_e_s * self.rate))

        return TimePoseSeries(
            self.t_sys_us[s_idx:e_idx],
            self.node_rots[s_idx:e_idx],
            self.node_ps[s_idx:e_idx],
        )

    def fix_time(self, t_21_us: int):
        self.node_t_us += t_21_us
        self.t_us_f0 += t_21_us
        self.t_sys_us += t_21_us

    def interpolate(self, t_new_us: NDArray):
        self.node_rots = slerp_rotation(
            self.node_rots, t_old_us=self.t_sys_us, t_new_us=t_new_us
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

    def save_csv(self, path: str | Path, using_opt: bool = False):
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
        ]

        t_us = self.node_t_us if not using_opt else self.opt_t_us
        ps = self.node_ps if not using_opt else self.opt_ps
        rots = self.node_rots if not using_opt else self.opt_rots

        data = np.concatenate(
            [
                t_us.reshape(-1, 1),
                ps,
                rots.as_quat(scalar_first=True),
            ],
            axis=1,
        )

        pd.DataFrame(data, columns=header).to_csv(
            path, index=False, float_format="%.8f"
        )
        print(f"Saved {len(self.node_ids)} poses to {path}")
