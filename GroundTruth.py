import argparse

from base import RTABData, UnitData

if __name__ == "__main__":
    default_dataset_path = "dataset/001/20251031_01_in/20251031_101025_SM-G9900"
    # 解析命令行参数，获取数据集路径
    arg_parser = argparse.ArgumentParser(description="Ground Truth Analysis")
    arg_parser.add_argument(
        "-u", "--unit", help="Path to the dataset unit", default=default_dataset_path
    )
    args = arg_parser.parse_args()
    unit_path: str = args.unit

    # 此处替换为自己的数据集路径
    fp = UnitData(unit_path)
    gt_data = RTABData(fp.gt_path)

    gt_data.save_csv(fp.target("rtab.csv"))
    gt_data.draw(save_path=fp.target("Trajectory.png"))
