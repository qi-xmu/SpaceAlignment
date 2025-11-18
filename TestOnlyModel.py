#! /Users/qi/Codespace/Android/NAVIO/app/src/main/cpp/SensorFusionAndroid/.venv/bin/python3
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=6, suppress=True)


class LoadModel:
    def __init__(
        self,
        model_path,
    ):
        self.device = torch.device("cpu")
        self.model = torch.jit.load(model_path)
        self.model.eval()
        pass

    def get_predict(self, block):
        inputs = torch.as_tensor(block, dtype=torch.float32, device=self.device)
        meas, meas_cov = self.model(inputs)
        output = meas.detach().numpy()
        return output


class LoadDataset:
    """
    加载数据集，使用AHRS转换
    """

    def __init__(self, imu_data_path, step=10, block_size=200, remove_gravity=False):
        self.step = step
        self.bs = block_size
        self.rm_g = remove_gravity

        self.data = pd.read_csv(imu_data_path).to_numpy()
        self.t_us = self.data[:, 0]
        self.gyro = self.data[:, 1:4]
        self.acce = self.data[:, 4:7]
        self.ahrs = self.data[:, 7:11]  # wxyz
        self.ahrs_xyzw = self.ahrs[:, [1, 2, 3, 0]]  # 转换为xyzw

    class AllanCalibration:
        pass

    @staticmethod
    def format_to_spec(t_us, acce, gyro, target_freq=200, allan_calibration=None):
        # 将 IMU 数据插值到目标频率
        out_t_us = np.arange(t_us[0], t_us[-1], 1e6 / target_freq)
        print(
            "Acce shape:",
            acce.shape,
            "Gyro shape:",
            gyro.shape,
            "Out_t shape:",
            out_t_us.shape,
        )
        f_acce = interp1d(t_us, acce, axis=0, fill_value="extrapolate")(out_t_us)
        f_gyro = interp1d(t_us, gyro, axis=0, fill_value="extrapolate")(out_t_us)
        return (out_t_us, f_acce, f_gyro)

    @staticmethod
    def _rotate_one(ahrs, vector):
        return R.from_quat(ahrs).apply(vector)

    def rotate(self):
        self.gyro_rotated = np.zeros_like(self.gyro)
        self.acce_rotated = np.zeros_like(self.acce)

        for i in range(len(self.ahrs)):
            self.gyro_rotated[i] = self._rotate_one(self.ahrs_xyzw[i], self.gyro[i])
            self.acce_rotated[i] = self._rotate_one(self.ahrs_xyzw[i], self.acce[i])
            if self.rm_g:
                self.acce_rotated[i] -= np.array([0, 0, 9.8])

        self.t_us, self.acce_rotated, self.gyro_rotated = self.format_to_spec(
            self.t_us, self.acce_rotated, self.gyro_rotated
        )
        print(
            f"Rotated IMU to world frame and resampled to {1e6 / (self.t_us[1] - self.t_us[0]):.2f} Hz"
        )
        self.imu_rotated = np.concatenate(
            (self.gyro_rotated, self.acce_rotated), axis=1
        )

    def save_rotated(self):
        # 保存文件 pandas
        imu_rotated_df = pd.DataFrame(
            self.imu_rotated,
            columns=["gyro_x", "gyro_y", "gyro_z", "acce_x", "acce_y", "acce_z"],
        )
        imu_rotated_df.insert(0, "timestamp_us", self.t_us)
        # 输出 8 位小数
        imu_rotated_df = imu_rotated_df.to_csv(
            "imu_rotated.csv", index=False, float_format="%.8f"
        )

    def get_block(self):
        i = 0
        while i + self.bs < len(self.imu_rotated):
            block = self.imu_rotated[i : i + self.bs]
            i += self.step
            yield block.T.reshape(1, 6, self.bs)


class ResultShow:
    def __init__(self, rate=20):
        self.interval = 1 / rate
        self.outs = []

    def add(self, out):
        self.outs.append(out)

    def show(self):
        array = np.array(self.outs)
        array = array.reshape(-1, 3)
        self.cum = np.cumsum(array * self.interval, axis=0)

        # 绘制 self.cum 的 xyz 坐标,优先绘制2d
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(111)
        ax1.plot(self.cum[:, 0], self.cum[:, 1], linewidth=1, label="Trajectory")
        ax1.scatter(
            *self.cum[0, :2], c="g", marker="o", s=100, label="Trajectory Start"
        )
        ax1.scatter(*self.cum[-1, :2], c="r", marker="x", s=100, label="Trajectory End")
        ax1.legend()
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_title("Trajectory")
        # xy等宽
        ax1.set_aspect("equal")
        ax1.grid(True)
        plt.show()


def main():
    model = LoadModel(
        "/Users/qi/Codespace/Android/NAVIO/app/src/main/cpp/SensorFusionAndroid/models/ZLX_03.pt"
    )
    data = LoadDataset("dataset/20251014_225509/imu.csv")

    print("Rotating IMU data...")
    data.rotate()
    data.save_rotated()  # 保存旋转后的数据

    # pause()

    res = ResultShow()
    for block in data.get_block():
        out = model.get_predict(block)
        print(out)
        res.add(out)
    res.show()


if __name__ == "__main__":
    main()
