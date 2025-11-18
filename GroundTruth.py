from base import RTABData, UnitData
from base.args_parser import DatasetArgsParser

if __name__ == "__main__":
    args = DatasetArgsParser().parse()
    assert args.unit is not None, "unit_path is required"
    unit_path: str = args.unit

    # 此处替换为自己的数据集路径
    fp = UnitData(unit_path)
    gt_data = RTABData(fp.gt_path)

    gt_data.save_csv(fp.target("rtab.csv"))
    gt_data.draw(save_path=fp.target("Trajectory.png"))
