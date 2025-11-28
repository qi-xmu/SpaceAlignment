from copy import copy

import cv2
import numpy as np
import rerun as rr
from numpy._typing import NDArray

import rerun_ext.rerun_calibration as rrec
from base.basetype import DataCheck, PoseSeries, Transform
from base.interpolate import get_time_series

from .datatype import (
    ARCoreData,
    CalibrationData,
    IMUData,
    RTABData,
    TimePoseSeries,
    UnitData,
)


class HandEyeAlg:
    Tsai = cv2.CALIB_HAND_EYE_TSAI
    Andreff = cv2.CALIB_HAND_EYE_ANDREFF
    Horaud = cv2.CALIB_HAND_EYE_HORAUD
    Park = cv2.CALIB_HAND_EYE_PARK
    Daniilidis = cv2.CALIB_HAND_EYE_DANIILIDIS


def invert_transform(R: NDArray, t: NDArray):
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def compose_transform(R1: NDArray, t1: NDArray, R2: NDArray, t2: NDArray):
    """T1 * T2"""
    R = R1 @ R2
    t = R1 @ t2 + t1
    return R, t


def _calibrate_T_gc(
    pose_bg: PoseSeries,
    pose_ct: PoseSeries,
    pose_cg: Transform | None = None,
    alg=HandEyeAlg.Park,
    rot_only=False,
) -> Transform:
    """calibrateHandEye return R_gc, t_gc
    隐含信息：base 和 target 为刚体，gripper 和 camera 为刚体。
    """
    assert len(pose_bg) > 10, f"Not enough data points {len(pose_bg)}"
    assert len(pose_bg) == len(pose_ct), f"{len(pose_bg)} != {len(pose_ct)}"
    # 标定
    if rot_only:
        pose_bg.reset_trans()
        pose_ct.reset_trans()

    R_gc, t_gc = cv2.calibrateHandEye(
        *pose_bg.get_series(), *pose_ct.get_series(), method=alg
    )
    rvec = cv2.Rodrigues(R_gc)[0]
    ang = np.linalg.norm(rvec)
    if ang != 0:
        rvec = rvec / ang
    print("旋转向量: ", rvec.flatten(), ang * 180 / np.pi)
    print("位移: ", t_gc.flatten())

    return Transform.from_raw(R_gc, t_gc.flatten())


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


def _calibrate_b1_b2(
    cs1: TimePoseSeries,
    cs2: TimePoseSeries,
    *,
    is_body_calc: bool = True,
    is_ref_calc: bool = True,
    rot_only=False,
):
    # 插值到相同频率
    t_new_us = get_time_series([cs1.t_us, cs2.t_us], rate=10)
    cs1 = cs1.interpolate(t_new_us)
    cs2 = cs2.interpolate(t_new_us)

    if is_body_calc:
        print("> 计算刚体之间的变换：")
        poses_rb1 = cs1.get_all()
        poses_br2 = cs2.get_all(inverse=True)
        pose_b12 = _calibrate_T_gc(poses_rb1, poses_br2, rot_only=rot_only)
        err_b1_b2 = calibrate_evaluate(
            pose_b12, poses_rb1, poses_br2, rot_only=rot_only
        )
    else:
        pose_b12 = Transform.identity()
        err_b1_b2 = None

    if is_ref_calc:
        print("> 计算参考坐标系之间的变换：")
        poses_br1 = cs1.get_all(inverse=True)
        poses_rb1 = cs2.get_all()
        pose_r12 = _calibrate_T_gc(poses_br1, poses_rb1, rot_only=rot_only)
        err_r1_r2 = calibrate_evaluate(
            pose_r12, poses_br1, poses_rb1, rot_only=rot_only
        )
    else:
        pose_r12 = Transform.identity()
        err_r1_r2 = None

    cd = CalibrationData(
        pose_b12,
        pose_r12,
        notes=f"err_sg_local = {err_b1_b2} err_sg_global = {err_r1_r2}",
    )
    return cd


def calibrate_evaluate(
    pose_gc: Transform, poses_bg: PoseSeries, poses_ct: PoseSeries, *, rot_only=False
):
    rot_errors = []
    trans_errors = []
    if rot_only:
        pose_gc.tran = np.zeros(3)
    for RA, tA, RB, tB in zip(
        *_get_A_B(*poses_bg.get_series(), *poses_ct.get_series())
    ):
        deg, trans = _transform_error(RA, tA, RB, tB, *pose_gc.get_raw())
        rot_errors.append(deg)
        trans_errors.append(trans)

    meas_err = float(np.mean(rot_errors)), float(np.mean(trans_errors))
    max_err = float(np.max(rot_errors)), float(np.max(trans_errors))

    print("平均误差 旋转（度）/ 平移（米）: {:.5f} / {:.5f}".format(*meas_err))
    print("最大误差 旋转（度）/ 平移（米）: {:.5f} / {:.5f}".format(*max_err))
    return meas_err, max_err


def calibrate_pose_series(
    cs_i: TimePoseSeries,
    cs_g: TimePoseSeries,
    cs_c: TimePoseSeries | None = None,
):
    if cs_c is not None:
        print("------------- 计算 Sensor - GT ")
        cd_cg = _calibrate_b1_b2(cs_c, cs_g, rot_only=False)
        return cd_cg
    else:
        cd = _calibrate_b1_b2(cs_i, cs_g, rot_only=True)
    return cd


def calibrate_unit(
    ud: UnitData,
    *,
    time_range: tuple = (None, None),
    using_rerun: bool = True,
    using_cam: bool = True,
    z_up: bool = True,
):
    print(f"Calibrating {ud.data_id}")
    imu_data = IMUData(ud.imu_path)
    gt_data = RTABData(ud.gt_path)
    dc = DataCheck.from_json(ud.check_file)
    print(f"使用时间差：{dc.t_gi_us}")
    gt_data.fix_time(dc.t_gi_us)

    cs_g = gt_data.get_time_pose_series(time_range)
    cs_i = imu_data.get_time_pose_series(time_range)

    if ud.using_cam and using_cam:
        print("Using Camera for calibration")
        cam_data = ARCoreData(ud.cam_path, z_up=z_up)
        cs_c = cam_data.get_time_pose_series(time_range)
        notes = "使用相机"

        cd = calibrate_pose_series(cs_i, cs_g, cs_c)
        if using_rerun:
            rrec.rerun_init(ud.data_id)
            imu_data.transform_to_world()
            rrec.send_imu_cam_data(imu_data, cam_data)
            rrec.send_pose_data(cs_g, cd)
    else:
        print("Not using Camera for calibration")
        cd = calibrate_pose_series(cs_i, cs_g)
        notes = "未使用相机，为标定位移"
        if using_rerun:
            rrec.rerun_init(ud.data_id)
            imu_data.transform_to_world()
            rrec.send_imu_cam_data(imu_data)
            rrec.send_pose_data(cs_g, cd)

    cd.to_json(ud.unit_calib_path, notes)
    # 如果是标定组内的数据，还会在组内生成标定文件
    if ud.is_calibr_data:
        cd.to_json(ud.group_calibr_path, notes)

    if using_rerun:
        rr.save(ud.target("data.rrd"))
    return cd
