from pathlib import Path

from base import load_calibration_data
from base.args_parser import DatasetArgsParser
from base.calibrate import calibrate_pose_series
from base.datatype import ARCoreData, IMUData, RTABData, UnitData
from rerun_ext import rerun_calibration as rrec


def view_only(path: Path | str):
    path = Path(path)
    unit = UnitData(path)
    gt_data = RTABData(unit.gt_path)
    imu_data = IMUData(unit.imu_path)
    cd = load_calibration_data(unit=unit)
    rrec.rerun_init("View IMU Data")
    rrec.send_imu_cam_data(imu_data)
    rrec.send_gt_data(gt_data, cd)


def view_calibration(path: Path | str):
    fp = UnitData(path)

    gt_data = RTABData(fp.gt_path)
    imu_data = IMUData(fp.imu_path)
    cam_data = ARCoreData(fp.cam_path, z_up=True)

    cs1 = imu_data.get_time_pose_series()
    cs2 = gt_data.get_time_pose_series()
    cs3 = cam_data.get_time_pose_series()
    cd, cd_ic = calibrate_pose_series(cs_i=cs1, cs_g=cs2, cs_c=cs3)

    imu_data.transform_to_world()
    rrec.rerun_init("View Trace")
    rrec.send_imu_cam_data(imu_data, cam_data, cd_ic)

    rrec.send_gt_data(gt_data, cd)


if __name__ == "__main__":
    args = DatasetArgsParser().parse()
    assert args.unit is not None, "Unit data path is required"
    # view_only(args.unit)
    view_calibration(args.unit)
