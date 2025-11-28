import rerun_ext.rerun_calibration as rrec
from base.action import dataset_action
from base.args_parser import DatasetArgsParser
from base.basetype import DataCheck
from base.datatype import GroupData, IMUData, NavioDataset, RTABData, UnitData

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
        dc = DataCheck.from_json(ud.check_file)
        imu_data = IMUData(ud.imu_path)
        gt_data = RTABData(ud.gt_path)
        gt_data.fix_time(dc.t_gi_us)

        rrec.rerun_init(ud.data_id)
        imu_data.world_gyro = imu_data.gyro
        imu_data.world_acce = imu_data.acce
        imu_data.transform_to_world()
        rrec.send_imu_cam_data(imu_data)

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
