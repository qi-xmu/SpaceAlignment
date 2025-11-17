from pathlib import Path

from base import ARCoreData, IMUData, RTABData, UnitData
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
    cam_data = ARCoreData(fp.cam_path, z_up=False)

    cs1 = imu_data.get_time_pose_series()
    cs2 = gt_data.get_time_pose_series()
    cs3 = cam_data.get_time_pose_series()

    # 标定结果，保存到文件
    cd32, _ = calibrate_b1_b2(
        cs_ref1_body1=cs3,
        cs_ref2_body2=cs2,
        rot_only=False,
    )

    cd13, _ = calibrate_b1_b2(
        cs_ref1_body1=cs1,
        cs_ref2_body2=cs3,
        rot_only=True,
    )

    imu_data.transform_to_world(qs=imu_data.ahrs_qs)
    rrec.rerun_init("View Trace")
    rrec.send_imu_cam_data(imu_data, cam_data, cd13)

    cd = cd32
    assert cd13.rot_ref_sensor_gt is not None
    assert cd32.rot_ref_sensor_gt is not None
    cd.rot_ref_sensor_gt = cd13.rot_ref_sensor_gt @ cd32.rot_ref_sensor_gt
    cd.tr_ref_sensor_gt = cd13.rot_ref_sensor_gt @ cd32.tr_ref_sensor_gt

    rrec.send_gt_data(gt_data, cd)


if __name__ == "__main__":
    # path = "dataset/001/20251031_01_in/Calibration/20251031_095725_SM-G9900"
    # path = "dataset/20251111_191453_SM-G9900"
    # path = "dataset/20251111_192622_SM-G9900"  # fail
    # path = "dataset/20251111_204152_SM-G9900"
    path = "dataset/001/20251031_01_in/20251031_101025_SM-G9900"
    # path = "dataset/001/20251031_01_in/20251031_102355_SM-G9900"
    # path = "dataset/001/20251031_01_in/20251031_115654_SM-G9900"
    # path = "dataset/001/20251031_01_in/20251031_103441_SM-G9900"
    view_only(path)
