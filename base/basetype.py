from typing import Literal

import numpy as np
from numpy.typing import NDArray

Time = NDArray[np.int64]
Pose = tuple[NDArray | None, NDArray | None]
Poses = tuple[list[NDArray], list[NDArray]]

SceneType = Literal["in", "out"]  # 场景类型，in 表示室内场景，out 表示室外场景
DeviceType = Literal[
    "SM-G9900",  # 三星 FE 21 5G
    "Redmi K30 Pro",  # 红米 k30 pro
    "ABR-AL60",  # 华为 Mate 60e,
    "Unknown",  # 未知设备
]
