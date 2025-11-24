from base.action import dataset_action
from base.args_parser import DatasetArgsParser
from base.datatype import GroupData, NavioDataset, UnitData

if __name__ == "__main__":
    # 解析命令行参数，获取数据集路径
    args = DatasetArgsParser()
    # time_range
    args.parser.add_argument(
        "--time_range", type=float, nargs=2, default=(0, 50), help="时间范围"
    )
    args.parse()
    unit_path = args.unit
    group_path = args.group
    dataset_path = args.dataset
    time_range = args.args.time_range
    visual = args.visual

    def action(ud: UnitData):
        with open(ud.unit_calib_path, "r", encoding="gbk") as f:
            content = f.read()
        print(content)
        with open(ud.unit_calib_path, "w", encoding="utf-8") as f:
            f.write(content)


    if unit_path:
        ud = UnitData(unit_path)
        action(ud)
    elif group_path:
        gd = GroupData(group_path)
        dataset_action(gd, action)
    elif dataset_path:
        ds = NavioDataset(dataset_path)
        dataset_action(ds, action)
    print("Analysis completed.")
