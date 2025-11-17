import sqlite3
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from .datatype import CalibrationData, TimePoseSeries


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
    def __init__(
        self, file_path: str | Path, calibr_data: CalibrationData | None = None
    ):
        self.file_path = str(file_path)
        self.calibr_data = calibr_data

        self.opt_ids: list[int] = []
        self.opt_qs: list[Quaternion] = []
        self.opt_ps: np.ndarray
        self.opt_t_us: np.ndarray

        self.node_t_us: np.ndarray
        self.t_us_f0: np.ndarray
        self.node_ids: list[int] = []
        self.node_qs: list[Quaternion] = []
        self.node_ps: np.ndarray

        if self.file_path.endswith(".csv"):
            print("Loading from CSV data...")
            self.load_csv_data()
        elif self.file_path.endswith(".db"):
            print("Loading from DB data...")
            # 从数据库中加载
            self.conn = sqlite3.connect(self.file_path)
            self.cursor = self.conn.cursor()
            self.load_node_data()
            self.load_opt_data()
        else:
            raise ValueError("RTAB-Map data file must be a .csv or .db file")

        self.t_us_f0 = self.node_t_us - self.node_t_us[0]
        self.t_sys_us = self.node_t_us
        self.node_freq = 1e6 / np.mean(np.diff(self.node_t_us))
        print(
            f"Loaded {len(self.node_ids)} nodes, {len(self.opt_ids)} optimized witch Freq: {self.node_freq} from the database. "
        )

    def load_csv_data(self):
        # #timestamp [us],p_RN_x [m],p_RN_y [m],p_RN_z [m],q_RN_w [],q_RN_x [],q_RN_y [],q_RN_z []
        data = pd.read_csv(self.file_path).to_numpy()
        self.node_t_us = data[:, 0]
        self.node_ps = data[:, 1:4]
        self.node_qs = [Quaternion(*q).unit for q in data[:, 4:]]
        self.opt_qs = self.node_qs
        self.opt_ps = self.node_ps

    def load_opt_data(self):
        # 查询Admin表中的opt_poses数据
        admin_opt_poses = self.cursor.execute(
            "SELECT opt_poses FROM Admin WHERE opt_poses IS NOT NULL"
        ).fetchone()
        decompressed_data = decompress_and_parse_data(admin_opt_poses[0])

        assert decompressed_data is not None, (
            "Failed to decompress admin_opt_poses data"
        )

        opt_ps = []
        for pose_values in struct.iter_unpack("12f", decompressed_data):
            pose_matrix = np.array(pose_values).reshape(3, 4)

            R = pose_matrix[:3, :3]
            p = pose_matrix[:3, 3]
            unit_q = rotation_matrix_to_quaternion(R).unit
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

        decompressed_ids = decompress_and_parse_data(admin_opt_ids[0])
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

        # 插值
        # cs = pose_interpolate(
        #     cs=self.get_time_pose_series(using_opt=True),
        #     t_new_us=self.node_t_us,
        # )
        # self.opt_t_us = cs.t_us
        # self.opt_qs = cs.qs
        # self.opt_ps = cs.ps

    def load_node_data(self):
        results = self.cursor.execute("""
            SELECT id, stamp, pose
            FROM Node
            WHERE pose IS NOT NULL
            ORDER BY id
        """).fetchall()

        assert results, "No node data found in the database."

        node_t_us = []
        node_ps = []
        for node_id, stamp, pose_blob in results:
            pose = unpack_pose_data(pose_blob)
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
        self, max_idx: int | None = None, *, using_opt: bool = False
    ) -> TimePoseSeries:
        return TimePoseSeries(
            # self.node_t_us  == self.t_sys_us
            ts=self.node_t_us[:max_idx] if not using_opt else self.opt_t_us[:max_idx],
            qs=self.node_qs[:max_idx] if not using_opt else self.opt_qs[:max_idx],
            ps=self.node_ps[:max_idx] if not using_opt else self.opt_ps[:max_idx],
        )

    def fix_time(self, t_21_us: int):
        t_21_us = int(t_21_us)
        self.node_t_us += t_21_us
        self.t_us_f0 += t_21_us
        self.opt_t_us += t_21_us

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
                plt.show()

            if save_path:
                fig.savefig(save_path, dpi=300)
        except ImportError:
            print("matplotlib is required for drawing the trajectory.")

    def save_csv(self, path: str | Path):
        # assert len(self.node_t_us) == len(self.opt_t_us), (
        #     f"Length mismatch: {len(self.node_t_us)} != {len(self.opt_t_us)}"
        # )
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


if __name__ == "__main__":
    path = "dataset/20251021_180034/251021-180058.db"
    rtab_data = RTABData(path)
    rtab_data.draw()
