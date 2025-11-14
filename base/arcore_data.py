import pandas as pd
import numpy as np
from pyquaternion import Quaternion
from .datatype import CalibrationSeries


class ARCoreData:
    def __init__(self, file_path, dataset_id=None, extend=True, z_up=False):
        self.file_path = file_path
        self.extend = extend
        self.z_up = z_up

        if self.z_up:
            self.R_SENSOR_TO_CAM = Quaternion()
        else:
            self.R_SENSOR_TO_CAM = Quaternion(axis=[1, 0, 0], angle=np.pi / 2)

        self.load_data()

    def __len__(self):
        return self.t_us.__len__()

    def _transform_world(self, ps):
        return np.einsum("ij,kj->ki", self.R_SENSOR_TO_CAM.rotation_matrix, ps)

    def _to_q_obj(self, qs):
        return [(self.R_SENSOR_TO_CAM * Quaternion(q)).unit for q in qs]

    def load_data(self):
        # #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []
        self.raw_data = pd.read_csv(self.file_path).to_numpy()

        self.t_us = self.raw_data[:, 0]
        self.ps = self._transform_world(self.raw_data[:, 1:4])
        self.qs = self.raw_data[:, 4:8]  # wxyz

        # convert to Quaternion object
        self.t_us_f0 = self.t_us - self.t_us[0]
        self.unit_qs_sensor = self._to_q_obj(self.qs)
        self.freq = 1e6 / np.mean(np.diff(self.t_us))

        self.extend = self.raw_data.shape[1] > 8
        if self.extend:
            self.ps_cam = self._transform_world(self.raw_data[:, 8:11])
            self.qs_cam = self.raw_data[:, 11:15]
            self.t_sys_us = self.raw_data[:, 15]

            self.ps_cam = (
                self.R_SENSOR_TO_CAM.rotation_matrix @ self.ps_cam.T).T

            self.unit_qs_cam = self._to_q_obj(self.qs_cam)

            # 根据 self.t_us 的时间间隔更新 self.t_sys_us，使其与 self.t_us 保持一致
            # 使用 t_us 的时间基准，但保持 t_sys_us 的起始时间
            self.t_sys_us = self.t_sys_us[0] + self.t_us_f0

    def get_calibr_series(self) -> CalibrationSeries:
        return CalibrationSeries(
            times=self.t_sys_us,
            rots=[q.rotation_matrix for q in self.unit_qs_sensor],
            trs=self.ps,
        )

    def draw(self, show=True):
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.ps[:, 0], self.ps[:, 1], label="ARCore Trajectory")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.grid()
            ax.set_aspect("equal", "box")
            ax.legend()
            if show:
                plt.show()
        except Exception as e:
            print(f"Error in drawing ARCore data: {e}")


if __name__ == "__main__":
    path = "/Users/qi/Codespace/Python/SpaceAlignment/dataset/20251014_225019/cam.csv"
    arcore_data = ARCoreData(path)
    arcore_data.draw()
