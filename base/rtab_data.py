from pathlib import Path
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import sqlite3
import zlib
import struct
from .datatype import CalibrationSeries


def decompress_and_parse_data(blob_data) -> bytes | None:
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


def rotation_matrix_to_quaternion(R):
    """将3x3旋转矩阵转换为四元数 (w, x, y, z)"""
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    if qw < 0:
        qw = -qw
        qx = -qx
        qy = -qy
        qz = -qz

    return Quaternion([qw, qx, qy, qz])


def unpack_pose_data(blob_data):
    """解压并解析RTAB-Map的pose数据"""
    decompressed_data = decompress_and_parse_data(blob_data)
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


class RTABData:
    def __init__(self, file_path: str | Path, id=None):
        self.id = id
        self.file_path = str(file_path)

        self.opt_ids: list[int] = []
        self.opt_ps: list[np.ndarray] = []
        self.opt_qs: list[Quaternion] = []
        self.opt_t_us:  np.ndarray
        self.node_t_us:  np.ndarray
        self.node_t_us_f0: np.ndarray
        self.node_ids: list[int] = []
        self.node_qs: list[Quaternion] = []
        self.node_ps: list[np.ndarray] = []

        assert self.file_path.endswith(
            ".db"), "RTAB-Map data file must be a .db file"

        # 从数据库中加载
        self.conn = sqlite3.connect(self.file_path)
        self.cursor = self.conn.cursor()
        self.load_node_data()
        self.load_opt_data()

        self.node_freq = 1e6 / np.mean(np.diff(self.node_t_us))

    def load_opt_data(self):
        # 查询Admin表中的opt_poses数据
        admin_opt_poses = self.cursor.execute(
            "SELECT opt_poses FROM Admin WHERE opt_poses IS NOT NULL"
        ).fetchone()
        decompressed_data = decompress_and_parse_data(admin_opt_poses[0])

        assert decompressed_data is not None, (
            "Failed to decompress admin_opt_poses data"
        )

        for pose_values in struct.iter_unpack("12f", decompressed_data):
            pose_matrix = np.array(pose_values).reshape(3, 4)

            p = pose_matrix[:3, 3]
            R = pose_matrix[:3, :3]
            unit_q = rotation_matrix_to_quaternion(R).unit
            self.opt_ps.append(p)
            self.opt_qs.append(unit_q)

        # 查询Admin表中的 opt_ids 数据
        admin_opt_ids = self.cursor.execute(
            "SELECT opt_ids FROM Admin WHERE opt_ids IS NOT NULL"
        ).fetchone()

        decompressed_ids = decompress_and_parse_data(admin_opt_ids[0])
        assert decompressed_ids is not None, "Failed to decompress admin_opt_ids data"

        self.opt_ids = list(
            struct.unpack(f"{len(decompressed_ids) // 4}i", decompressed_ids)
        )
        assert (
            self.opt_ids.__len__() == self.opt_ps.__len__() == self.opt_qs.__len__()
        ), "Mismatch in lengths of opt_ids, ps, and unit_qs"

        # 通过 ids 获取时间戳
        opt_t_us = []
        for opt_id in self.opt_ids:
            t = self.node_t_us[self.node_ids.index(opt_id)]
            opt_t_us.append(t)

        self.opt_t_us = np.array(opt_t_us)
        print(f"Loaded {len(self.opt_ids)} optimized poses from the database.")

    def load_node_data(self):
        results = self.cursor.execute("""
            SELECT id, stamp, pose
            FROM Node
            WHERE pose IS NOT NULL
            ORDER BY id
        """).fetchall()

        assert results, "No node data found in the database."

        node_t_us = []
        for node_id, stamp, pose_blob in results:
            pose = unpack_pose_data(pose_blob)
            if pose is None:
                continue

            p, unit_q = pose
            self.node_ids.append(node_id)
            node_t_us.append(int(stamp * 1e6))  # convert to us
            self.node_ps.append(p)
            self.node_qs.append(unit_q)

        self.node_t_us = np.array(node_t_us)
        self.node_t_us_f0 = self.node_t_us - self.node_t_us[0]
        self.node_freq = 1e6 / np.mean(np.diff(self.node_t_us))
        print(
            f"Loaded {len(self.node_ids)} nodes witch Freq: {self.node_freq} from the database."
        )

    def get_calibr_series(self) -> CalibrationSeries:
        return CalibrationSeries(
            times=self.node_t_us,
            rots=[q.rotation_matrix for q in self.node_qs],
            trs=self.node_ps
        )

    def save_data(self, name, data):
        pass

    def draw(self, show=True, save_path=None):
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # 绘制开始点和结束点
            ps = np.array(self.opt_ps)
            ax.plot(ps[0, 0], ps[0, 1], "o", color="g", label="Start")
            ax.plot(ps[-1, 0], ps[-1, 1], "x", color="r", label="End")
            ax.plot(ps[:, 0], ps[:, 1],
                    label="Optimized Trajectory", color="b")

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

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"{self.id} Trajectory")
            ax.grid(True)
            ax.set_aspect("equal", "box")
            # 设置 字体大小
            ax.legend(fontsize=8)
            if show:
                plt.show()

            if save_path:
                plt.savefig(save_path, dpi=300)
        except ImportError:
            print("matplotlib is required for drawing the trajectory.")

    def save_csv(self, path: str | Path):
        # #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],
        header = [
            "#timestamp [us]",
            "p_RS_R_x [m]",
            "p_RS_R_y [m]",
            "p_RS_R_z [m]",
            "q_RS_w []",
            "q_RS_x []",
            "q_RS_y []",
            "q_RS_z []",
        ]
        qs = np.array([q.elements for q in self.node_qs])

        data = np.concatenate([self.node_t_us_f0.reshape(-1, 1),
                               self.node_ps, qs], axis=1)

        pd.DataFrame(data, columns=header).to_csv(
            path, index=False, float_format="%.8f")
        print(f"Saved {len(self.node_ids)} poses to {path}")


if __name__ == "__main__":
    path = "dataset/20251021_180034/251021-180058.db"
    rtab_data = RTABData(path)
    rtab_data.draw()
