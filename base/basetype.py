import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

Time = NDArray[np.int64]
Pose = tuple[NDArray | None, NDArray | None]
Poses = tuple[NDArray, NDArray]

SceneType = Literal["in", "out"]  # 场景类型，in 表示室内场景，out 表示室外场景
DeviceType = Literal[
    "SM-G9900",  # 三星 FE 21 5G
    "Redmi K30 Pro",  # 红米 k30 pro
    "ABR-AL60",  # 华为 Mate 60e,
    "Unknown",  # 未知设备
]


@dataclass
class Transform:
    rot: Rotation
    tran: NDArray

    def __init__(self, rot: Rotation, tran: NDArray):
        self.rot = rot
        self.tran = tran

    def __mul__(self, other: Self):
        return Transform(self.rot * other.rot, self.rot.apply(other.tran) + self.tran)

    def get_raw(self):
        return self.rot.as_matrix(), self.tran

    @classmethod
    def from_raw(cls, raw_rot: NDArray, trans: NDArray):
        return cls(Rotation.from_matrix(raw_rot), trans)

    @classmethod
    def identity(cls):
        return cls(Rotation.identity(), np.zeros(3))

    def inverse(self):
        return Transform(self.rot.inv(), -self.rot.inv().apply(self.tran))


@dataclass
class PoseSeries:
    rots: Rotation
    trans: NDArray

    def inverse(self):
        return PoseSeries(self.rots.inv(), -self.rots.inv().apply(self.trans))

    def __len__(self):
        return len(self.rots)

    def get_series(self):
        return [rot.as_matrix() for rot in self.rots], [tran for tran in self.trans]

    def reset_trans(self):
        self.trans = np.zeros((len(self.rots), 3))


@dataclass
class DataCheck:
    t_gi_us: int

    @staticmethod
    def from_json(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "check_time_diff" in data
            check_time_diff = data["check_time_diff"]
            assert "time_diff_21_us" in check_time_diff
            t_gi_us = check_time_diff["time_diff_21_us"]
            return DataCheck(t_gi_us)
