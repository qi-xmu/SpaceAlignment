from base import RTABData, UnitPath
import numpy as np
import sys


# 解析命令行参数，获取数据集路径
if len(sys.argv) < 2:
    print("Usage: python GroundTruth.py <dataset_path>")
    sys.exit(1)
    
dataset_path = sys.argv[1]
# 此处替换为自己的数据集路径
fp = UnitPath(dataset_path)
gt_data = RTABData(fp.gt_path, id=fp.base_dir)

# 检查真值的频率
ts = gt_data.node_t_us
print(f"Freq: {1e6 / np.mean(np.diff(ts))} Hz")
# 最大时间差距
print(f"Max Time Diff: {np.max(np.diff(ts)) * 1e-6} s")
# 最小时间差距
print(f"Min Time Diff: {np.min(np.diff(ts)) * 1e-6} s")

gt_data.draw(save_path=fp.target("Trajectory.png"))
