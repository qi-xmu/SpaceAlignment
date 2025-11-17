import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from .datatype import TimePoseSeries


class ARCoreData:
    def __init__(self, file_path, dataset_id=None, extend=True, z_up=False):
        self.file_path = file_path
        self.extend = extend
        self.z_up = z_up

        if self.z_up:
            self.base_sensor_cam = Quaternion()
        else:
            self.base_sensor_cam = Quaternion(axis=[1, 0, 0], angle=np.pi / 2)

        self.load_data()

    def __len__(self):
        return self.sensor_t_us.__len__()

    def _transform_world(self, ps):
        return np.einsum("ij,kj->ki", self.base_sensor_cam.rotation_matrix, ps)

    def _to_q_obj(self, qs):
        return [(self.base_sensor_cam * Quaternion(q)).unit for q in qs]

    def load_data(self):
        # #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []
        self.raw_data = pd.read_csv(self.file_path).to_numpy()

        self.sensor_t_us = self.raw_data[:, 0]
        self.raw_sensor_ps = self.raw_data[:, 1:4]
        self.raw_sensor_qs = self.raw_data[:, 4:8]  # wxyz

        # convert to Quaternion object
        self.t_us_f0 = self.sensor_t_us - self.sensor_t_us[0]
        self.sensor_qs = self._to_q_obj(self.raw_sensor_qs)
        self.sensor_ps = self._transform_world(self.raw_sensor_ps)
        self.freq = 1e6 / np.mean(np.diff(self.sensor_t_us))

        self.extend = self.raw_data.shape[1] > 8
        if self.extend:
            self.raw_cam_ps = self.raw_data[:, 8:11]
            self.raw_cam_qs = self.raw_data[:, 11:15]
            self.system_t_us = self.raw_data[:, 15]

            self.cam_ps = self._transform_world(self.raw_cam_ps)
            self.cam_qs = self._to_q_obj(self.raw_cam_qs)
            # 根据 self.t_us 的时间间隔更新 self.t_sys_us，使其与 self.t_us 保持一致
            # 使用 t_us 的时间基准，但保持 t_sys_us 的起始时间
            self.system_t_us = self.system_t_us[0] + self.t_us_f0

    def get_time_pose_series(
        self, max_idx: int | None = None, *, using_cam: bool = False
    ) -> TimePoseSeries:
        return TimePoseSeries(
            ts=self.system_t_us[:max_idx],
            qs=self.sensor_qs[:max_idx] if not using_cam else self.cam_qs[:max_idx],
            ps=self.sensor_ps[:max_idx] if not using_cam else self.cam_ps[:max_idx],
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
