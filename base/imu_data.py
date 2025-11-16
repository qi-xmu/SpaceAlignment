from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from .datatype import TimePoseSeries


@final
class IMUData:
    def __init__(self, file_path: str | Path, target_freq: int = 200) -> None:
        self.file_path = str(file_path)
        self.target_freq = target_freq
        self.raw_data = np.array([])
        self.t_us = np.array([])
        self.gyro = np.array([])
        self.acce = np.array([])
        self.raw_ahrs = np.array([])
        self.ahrs_qs: list[Quaternion] = []
        self.t_us_f0 = np.array([])
        self.t_sys_us = np.array([])
        self.imu_freq: float = 0.0
        self.load_data()

    def load_data(self) -> None:
        """Load IMU data from CSV file."""
        # timestamp [us],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],
        # a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2],
        # q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []
        df: pd.DataFrame = pd.read_csv(self.file_path)
        self.raw_data = df.to_numpy()

        self.t_us = self.raw_data[:, 0]
        self.gyro = self.raw_data[:, 1:4]  # angular velocity
        self.acce = self.raw_data[:, 4:7]  # linear acceleration
        self.raw_ahrs = self.raw_data[:, 7:11]  # orientation

        # Convert quaternions to unit quaternions
        self.ahrs_qs = [Quaternion(q).unit for q in self.raw_ahrs]

        if len(self.t_us) > 1:
            self.t_us_f0 = self.t_us - self.t_us[0]

        self.extend = bool(self.raw_data.shape[1] > 11)
        if self.extend:
            self.t_sys_us = self.raw_data[:, 11]  # 1970 us
            self.t_sys_us = self.t_sys_us[0] + self.t_us_f0

        # Calculate IMU frequency
        if len(self.t_us) > 1:
            time_diffs = np.diff(self.t_us)
            self.imu_freq = float(1e6 / np.mean(time_diffs))
            print(f"IMU frequency: {self.imu_freq:.2f} Hz")
        else:
            print("Warning: Not enough data points to calculate frequency")

    def get_time_pose_series(self, max_idx: int | None = None) -> TimePoseSeries:
        return TimePoseSeries(
            ts=self.t_sys_us[:max_idx],
            qs=self.ahrs_qs[:max_idx],
            ps=np.zeros((len(self.ahrs_qs[:max_idx]), 3)),
        )

    def transform_to_world(
        self, *, rots: np.ndarray | None = None, qs: list[Quaternion] | None = None
    ):
        if rots is None and qs is not None:
            rots = np.array([q.rotation_matrix for q in qs])

        self.world_acce = np.einsum("ijk,ik->ij", rots, self.acce)  # type: ignore
        self.world_gyro = np.einsum("ijk,ik->ij", rots, self.gyro)  # type: ignore

    def format_to_spec(self, target_freq: int = 200) -> None:
        """Format data to specified frequency (placeholder method)."""
        print(f"Formatting to target frequency: {target_freq} Hz")
        # Implementation would go here


if __name__ == "__main__":
    path = "/Users/qi/Codespace/Python/SpaceAlignment/dataset/RedmiK30PRO/2025-728-093146/imu.csv"
    imu_data = IMUData(path)
