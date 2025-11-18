from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyquaternion import Quaternion

from .datatype import Time, TimePoseSeries
from .interpolate import interpolate_vector3d, slerp_quaternion


@final
class IMUData:
    t_us: Time
    t_us_f0: Time
    t_sys_us: Time
    raw_ahrs: NDArray
    acce: NDArray
    gyro: NDArray
    rate: float

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = str(file_path)
        self.ahrs_qs: list[Quaternion] = []
        self.load_data()

    def __len__(self):
        assert len(self.acce) == len(self.gyro) == len(self.t_sys_us)
        return len(self.t_sys_us)

    def load_data(self) -> None:
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
            self.t_sys_us = self.t_us

        # Calculate IMU frequency
        if len(self.t_us) > 1:
            time_diffs = np.diff(self.t_us)
            self.rate = float(1e6 / np.mean(time_diffs))
            print(f"IMU frequency: {self.rate:.2f} Hz")
        else:
            print("Warning: Not enough data points to calculate frequency")

    def get_time_pose_series(self, max_idx: int | None = None) -> TimePoseSeries:
        return TimePoseSeries(
            t_us=self.t_sys_us[:max_idx],
            qs=self.ahrs_qs[:max_idx],
            ps=np.zeros((len(self.ahrs_qs[:max_idx]), 3)),
        )

    def interpolate(self, t_new_us: np.ndarray) -> None:
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
        self, *, rots: np.ndarray | None = None, qs: list[Quaternion] | None = None
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

    def format_to_spec(self, target_freq: int = 200) -> None:
        """Format data to specified frequency (placeholder method)."""
        print(f"Formatting to target frequency: {target_freq} Hz")
        # Implementation would go here


if __name__ == "__main__":
    path = "/Users/qi/Codespace/Python/SpaceAlignment/dataset/RedmiK30PRO/2025-728-093146/imu.csv"
    imu_data = IMUData(path)
