import argparse

import numpy as np

from base import RTABData, UnitData

if __name__ == "__main__":
    default_dataset_path = "dataset/001/20251031_01_in/20251031_101025_SM-G9900"
    # 解析命令行参数，获取数据集路径
    arg_parser = argparse.ArgumentParser(description="Ground Truth Analysis")
    arg_parser.add_argument(
        "-d",
        "--dataset",
        help="Path to the dataset",
        default=default_dataset_path,
    )
    args = arg_parser.parse_args()

    dataset_path: str = args.dataset

    # 此处替换为自己的数据集路径
    fp = UnitData(dataset_path)
    gt_data = RTABData(fp.gt_path)

    # 检查真值的频率
    ts = gt_data.node_t_us
    print(f"Freq: {1e6 / np.mean(np.diff(ts))} Hz")
    # 最大时间差距
    print(f"Max Time Diff: {np.max(np.diff(ts)) * 1e-6} s")
    # 最小时间差距
    print(f"Min Time Diff: {np.min(np.diff(ts)) * 1e-6} s")

    gt_data.save_csv(fp.target("rtab.csv"))
    gt_data.draw(save_path=fp.target("Trajectory.png"))
