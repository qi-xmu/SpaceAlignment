from pathlib import Path

import cv2
import numpy as np
import rerun as rr

import rerun_ext.rerun_calibration as rrec
from base import (
    ARCoreData,
    CalibrationData,
    GroupData,
    IMUData,
    Pose,
    Poses,
    RTABData,
    TimePoseSeries,
    UnitData,
)
from base.interpolate import get_time_series, pose_interpolate
from time_diff import match_correlation


class HandEyeAlg:
    Tsai = cv2.CALIB_HAND_EYE_TSAI
    Andreff = cv2.CALIB_HAND_EYE_ANDREFF
    Horaud = cv2.CALIB_HAND_EYE_HORAUD
    Park = cv2.CALIB_HAND_EYE_PARK
    Daniilidis = cv2.CALIB_HAND_EYE_DANIILIDIS


def invert_transform(R, t) -> tuple[np.ndarray, np.ndarray]:
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def compose_transform(R1, t1, R2, t2) -> tuple[np.ndarray, np.ndarray]:
    """T1 * T2"""
    R = R1 @ R2
    t = R1 @ t2 + t1
    return R, t


def calibrate_pose_gripper_camera(
    poses_base_gripper: Poses,
    poses_camera_target: Poses,
    pose_camera_gripper=(None, None),
    alg=HandEyeAlg.Park,
    rot_only=False,
) -> Pose:
    """calibrateHandEye return R_gc, t_gc
    隐含信息：base 和 target 为刚体，gripper 和 camera 为刚体。
    """
    assert len(poses_base_gripper[0]) == len(poses_camera_target[0])
    # 标定
    if rot_only:
        poses_base_gripper = (
            poses_base_gripper[0],
            [np.zeros(3) for _ in range(len(poses_base_gripper[1]))],
        )
        poses_camera_target = (
            poses_camera_target[0],
            [np.zeros(3) for _ in range(len(poses_camera_target[1]))],
        )

    R_gc, t_gc = cv2.calibrateHandEye(
        *poses_base_gripper,
        *poses_camera_target,
        *pose_camera_gripper,
        method=alg,
    )
    rvec = cv2.Rodrigues(R_gc)[0]
    ang = np.linalg.norm(rvec)
    if ang != 0:
        rvec = rvec / ang
    print("旋转向量: ", rvec.flatten(), ang * 180 / np.pi)
    print("位移: ", t_gc.flatten())
    return R_gc, t_gc.flatten()


# %% 误差评估
def _get_A_B(R_bg, t_bg, R_ct, t_ct):
    RA_list = []
    tA_list = []
    RB_list = []
    tB_list = []
    n = len(R_bg)
    for i in range(n - 1):
        # 1. 机械臂运动A: A_i = Tgb_{i+1}^{-1} * Tgb_{i}
        Rgb_i = R_bg[i]
        tgb_i = t_bg[i]
        Rgb_next = R_bg[i + 1]
        tgb_next = t_bg[i + 1]
        Rgb_next_inv, tgb_next_inv = invert_transform(Rgb_next, tgb_next)
        RA, tA = compose_transform(Rgb_next_inv, tgb_next_inv, Rgb_i, tgb_i)
        RA_list.append(RA)
        tA_list.append(tA)

        # 2. 标定板运动B: B_i = Ttc_{i+1} * Ttc_{i}^{-1}
        Rtc_i = R_ct[i]
        ttc_i = t_ct[i]
        Rtc_next = R_ct[i + 1]
        ttc_next = t_ct[i + 1]
        Rtc_i_inv, ttc_i_inv = invert_transform(Rtc_i, ttc_i)
        RB, tB = compose_transform(Rtc_next, ttc_next, Rtc_i_inv, ttc_i_inv)
        RB_list.append(RB)
        tB_list.append(tB)
    return RA_list, tA_list, RB_list, tB_list


def _transform_error(RA, tA, RB, tB, RX, tX):
    # A * X
    R_AX = RA @ RX
    t_AX = RA @ tX + tA
    # X * B
    R_XB = RX @ RB
    t_XB = RX @ tB + tX
    # 旋转误差
    rot_err_rad = np.arccos(np.clip((np.trace(R_AX.T @ R_XB) - 1) / 2, -1, 1))
    rot_err_deg = np.degrees(rot_err_rad)
    # 平移误差
    trans_err = np.linalg.norm(t_AX - t_XB)
    return rot_err_deg, trans_err


def calibrate_b1_b2(
    *,
    cs_ref1_body1: TimePoseSeries,
    cs_ref2_body2: TimePoseSeries,
    calibr_data: CalibrationData = CalibrationData(),
    is_body_calc: bool = True,
    is_ref_calc: bool = True,
    is_t_diff=True,
    show_t_diff=False,
    rot_only=False,
):
    # 修复时间差距
    t21_us = match_correlation(cs_ref1_body1, cs_ref2_body2)
    cs_ref2_body2.t_us += t21_us
    # 插值到相同频率
    t_new_us = get_time_series([cs_ref1_body1.t_us, cs_ref2_body2.t_us], rate=2)
    cs1 = pose_interpolate(cs=cs_ref1_body1, t_new_us=t_new_us)
    cs2 = pose_interpolate(cs=cs_ref2_body2, t_new_us=t_new_us)

    if is_body_calc:
        print("> 计算刚体之间的变换：")
        poses_ref1_body1 = cs1.get_all()
        poses_body2_ref2 = cs2.get_all(inverse=True)
        pose_body1_body2 = calibrate_pose_gripper_camera(
            poses_ref1_body1,
            poses_body2_ref2,
            (calibr_data.rot_sensor_gt, calibr_data.tr_sensor_gt),
            rot_only=rot_only,
        )
        err_b1_b2 = evaluate(
            pose_body1_body2, poses_ref1_body1, poses_body2_ref2, rot_only=rot_only
        )
        poses_ref1_body1, poses_body2_ref2 = None, None
    else:
        pose_body1_body2 = (None, None)
        err_b1_b2 = None

    if is_ref_calc:
        print("> 计算参考坐标系之间的变换：")
        poses_body1_ref1 = cs1.get_all(inverse=True)
        poses_ref2_body2 = cs2.get_all()
        pose_ref1_ref2 = calibrate_pose_gripper_camera(
            poses_body1_ref1,
            poses_ref2_body2,
            (calibr_data.rot_ref_sensor_gt, calibr_data.tr_ref_sensor_gt),
            rot_only=rot_only,
        )
        err_r1_r2 = evaluate(
            pose_ref1_ref2, poses_body1_ref1, poses_ref2_body2, rot_only=rot_only
        )
        poses_body1_ref1, poses_ref2_body2 = None, None
    else:
        pose_ref1_ref2 = (None, None)
        err_r1_r2 = None

    cd = CalibrationData(
        *pose_body1_body2,
        *pose_ref1_ref2,
        err_b1_b2,
        err_r1_r2,
    )
    return cd


def evaluate(pose_gc: Pose, poses_bg, poses_ct, *, rot_only=False):
    rot_errors = []
    trans_errors = []
    if rot_only:
        pose_gc = pose_gc[0], np.zeros(3)
    for RA, tA, RB, tB in zip(*_get_A_B(*poses_bg, *poses_ct)):
        deg, trans = _transform_error(RA, tA, RB, tB, *pose_gc)
        rot_errors.append(deg)
        trans_errors.append(trans)

    meas_err = np.mean(rot_errors), np.mean(trans_errors)
    max_err = np.max(rot_errors), np.max(trans_errors)

    print("平均误差 旋转（度）/ 平移（米）: {:.5f} / {:.5f}".format(*meas_err))
    print("最大误差 旋转（度）/ 平移（米）: {:.5f} / {:.5f}".format(*max_err))
    return meas_err, max_err


def calibrate_sensor_camera(cam_data: ARCoreData):
    cs_sensor = cam_data.get_time_pose_series(using_cam=False)
    cs_camera = cam_data.get_time_pose_series(using_cam=True)
    print("--------------- 计算 Sensor - Camera")
    cs_sc = calibrate_b1_b2(
        cs_ref1_body1=cs_sensor,
        cs_ref2_body2=cs_camera,
    )
    return cs_sc


def calibrate_pose_series(
    *,
    imu_series: TimePoseSeries,
    gt_series: TimePoseSeries,
    cam_series: TimePoseSeries | None = None,
):
    if cam_series is not None:
        print("------------- 计算 Sensor - GT ")
        cd_cg = calibrate_b1_b2(
            cs_ref1_body1=cam_series, cs_ref2_body2=gt_series, rot_only=False
        )
        print("------------- 计算 AHRS - Sensor")
        cd_ic = calibrate_b1_b2(
            cs_ref1_body1=imu_series, cs_ref2_body2=cam_series, rot_only=True
        )
        cd = cd_cg
        assert cd_ic.rot_ref_sensor_gt is not None
        # 公式 R_ig = R_ic @ R_cg, t_ig = R_ic @ t_cg
        cd.rot_ref_sensor_gt = cd_ic.rot_ref_sensor_gt @ cd_cg.rot_ref_sensor_gt
        cd.tr_ref_sensor_gt = cd_ic.rot_ref_sensor_gt @ cd_cg.tr_ref_sensor_gt
        return cd, cd_ic
    else:
        cd = calibrate_b1_b2(
            cs_ref1_body1=imu_series,
            cs_ref2_body2=gt_series,
            rot_only=True,
        )
    return cd, CalibrationData()


def calibrate_unit(
    path: Path | str,
    *,
    using_rerun: bool = True,
):
    print(f"Calibrating {path}")
    path = Path(path)
    unit = UnitData(path)
    imu_data = IMUData(unit.imu_path)
    gt_data = RTABData(unit.gt_path)

    cs_i = imu_data.get_time_pose_series()
    cs_g = gt_data.get_time_pose_series()

    if unit.using_cam:
        cam_data = ARCoreData(unit.cam_path, z_up=unit.is_z_up)
        cs_c = cam_data.get_time_pose_series()
        notes = "使用相机"

        cd, cd_ic = calibrate_pose_series(
            imu_series=cs_i,
            gt_series=cs_g,
            cam_series=cs_c,
        )
        if using_rerun:
            rrec.rerun_init(path.name)
            rrec.send_imu_cam_data(imu_data, cam_data, cd_ic)
            rrec.send_gt_data(gt_data, cd)
    else:
        cd, _ = calibrate_pose_series(
            imu_series=cs_i,
            gt_series=cs_g,
        )
        notes = "未使用相机，为标定位移"
        if using_rerun:
            rrec.rerun_init(path.name)
            rrec.send_imu_cam_data(imu_data)
            rrec.send_gt_data(gt_data, cd)

    cd.to_json(unit.calibr_path, notes)
    if using_rerun:
        rr.save(unit.target("data.rrd"))
    return cd


def calibrate_group(path):
    gp = GroupData(path)
    for unit in gp.raw_calibr_path.iterdir():
        if unit.is_dir():
            calibrate_unit(unit)


def load_calibration_data(
    *,
    unit: UnitData,
    using_rerun: bool = False,
):
    # 加载 校准数据
    try:
        cd = CalibrationData.from_json(unit.calibr_path)
    except Exception as _:
        print("-" * 20, f"标定 {unit.device_name}")
        cd = calibrate_unit(unit.base_dir, using_rerun=using_rerun)
        cd.to_json(unit.calibr_path)
    return cd
