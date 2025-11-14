import os
import json
import numpy as np
import cv2
from base import CalibrationSeries, Pose
import TimeMatch


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
    poses_base_gripper,
    poses_camera_target,
    pose_camera_gripper=(None, None),
    alg=HandEyeAlg.Park
) -> tuple[np.ndarray, np.ndarray]:
    """calibrateHandEye return R_gc, t_gc
    隐含信息：base 和 target 为刚体，gripper 和 camera 为刚体。
    """
    # 标定
    R_gc, t_gc = cv2.calibrateHandEye(
        *poses_base_gripper,
        *poses_camera_target,
        *pose_camera_gripper,
        method=alg,
    )
    print("旋转矩阵:\n", R_gc)
    print("位移:\n", t_gc.flatten())
    return R_gc, t_gc


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
    cs_ref1_body1: CalibrationSeries,
    cs_ref2_body2: CalibrationSeries,
    result_path=None
):
    ts1 = cs_ref1_body1.times
    ts2 = cs_ref2_body2.times

    # 时间匹配，生成 索引对 (idx_ts1, idx_ts2, time_diff_us)
    matches = TimeMatch.match(ts1, ts2)

    poses_ref1_body1 = cs_ref1_body1.get_match(matches, index=0)
    poses_body2_ref2 = cs_ref2_body2.get_match(matches, index=1, inverse=True)
    pose_body1_body2 = calibrate_pose_gripper_camera(
        poses_ref1_body1, poses_body2_ref2)
    err_b1_b2 = evaluate(pose_body1_body2, poses_ref1_body1, poses_body2_ref2)
    poses_ref1_body1, poses_body2_ref2 = None, None

    poses_body1_ref1 = cs_ref1_body1.get_match(matches, index=0, inverse=True)
    poses_ref2_body2 = cs_ref2_body2.get_match(matches, index=1)
    pose_ref1_ref2 = calibrate_pose_gripper_camera(
        poses_body1_ref1, poses_ref2_body2)
    err_r1_r2 = evaluate(pose_ref1_ref2, poses_body1_ref1, poses_ref2_body2)
    poses_body1_ref1, poses_ref2_body2 = None, None

    if result_path:
        calibrate_json(
            result_path,
            pose_body1_body2,
            err_b1_b2,
            pose_ref1_ref2,
            err_r1_r2
        )
    return pose_body1_body2, pose_ref1_ref2, matches


def evaluate(pose_gc, poses_bg, poses_ct):
    rot_errors = []
    trans_errors = []
    for RA, tA, RB, tB in zip(*_get_A_B(*poses_bg, *poses_ct)):
        deg, trans = _transform_error(RA, tA, RB, tB,  *pose_gc)
        rot_errors.append(deg)
        trans_errors.append(trans)

    meas_err = np.mean(rot_errors), np.mean(trans_errors)
    max_err = np.max(rot_errors), np.max(trans_errors)

    print("旋转误差（度）: {:.4f} 平移误差（米）: {:.5f}".format(*meas_err))
    print("最大旋转误差（度）: {:.4f} 最大平移误差（米）: {:.5f}".format(*max_err))
    return meas_err, max_err


def calibrate_json(result_file: str, pose_b1_b2: Pose, err_b1_b2, pose_r1_r2: Pose, err_r1_r2, is_save=True):
    # 检查文件是否存在，如果存在，则读取 notes
    notes = """Pose: Sensor_Groundtruth
Error: [mean_rot_err, mean_trans_err, max_rot_err, max_trans_err]
"""
    # 准备保存的数据
    result_data = {
        "dataset": str(result_file),
        "rot_sensor_gt": pose_b1_b2[0].tolist(),
        "trans_sensor_gt": pose_b1_b2[1].flatten().tolist(),
        "rot_sensor_gt_err": list(err_b1_b2),
        "rot_ref_sensor_gt": pose_r1_r2[0].tolist(),
        "trans_ref_sensor_gt": pose_r1_r2[1].flatten().tolist(),
        "rot_ref_sensor_gt_err": list(err_r1_r2),
        "notes": notes,
    }
    if is_save:
        # 保存到JSON文件
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump([result_data], f, indent=4, ensure_ascii=False)
        print(f"标定结果已保存到: {result_file}")

    return result_data


if __name__ == "__main__":
    pass
