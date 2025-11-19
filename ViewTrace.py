from pathlib import Path

from base import ARCoreData, IMUData, RTABData, UnitData
from base.args_parser import DatasetArgsParser
from hand_eye import calibrate_b1_b2, load_calibration_data
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

    cs1 = imu_data.get_time_pose_series(int(imu_data.rate * 20))
    cs2 = gt_data.get_time_pose_series(int(gt_data.rate * 20))
    cs3 = cam_data.get_time_pose_series(int(cam_data.rate * 20))
    cd_ic = calibrate_b1_b2(cs1, cs3, rot_only=True)
    cd = calibrate_b1_b2(cs1, cs2, rot_only=True)

    imu_data.transform_to_world()
    rrec.rerun_init("View Trace")
    rrec.send_imu_cam_data(imu_data, cam_data, cd_ic)
    rrec.send_gt_data(gt_data, cd)


if __name__ == "__main__":
    args = DatasetArgsParser().parse()
    assert args.unit is not None, "Unit data path is required"
    # view_only(args.unit)
    view_calibration(args.unit)
