from base import Dataset, IMUData, RTABData
from base.arcore_data import ARCoreData
from base.interpolate import get_time_series, interpolate_vector3d, pose_interpolate

if __name__ == "__main__":
    # fp = FilePath("/Users/qi/Resources/Dataset001")

    exit()
    flatten = fp.flatten()
    flatten0 = flatten[0]

    imu_data = IMUData(flatten0.imu_path)
    cam_data = ARCoreData(flatten0.cam_path)
    gt_data = RTABData(flatten0.gt_path)

    t_new_us = get_time_series([imu_data.t_sys_us, gt_data.t_sys_us])

    cs_gt = pose_interpolate(
        cs=gt_data.get_time_pose_series(),
        t_new_us=t_new_us,
    )
    imu_data.acce = interpolate_vector3d(
        vec3d=imu_data.acce,
        t_old_us=imu_data.t_sys_us,
        t_new_us=t_new_us,
    )
    imu_data.gyro = interpolate_vector3d(
        vec3d=imu_data.gyro,
        t_old_us=imu_data.t_sys_us,
        t_new_us=t_new_us,
    )
    imu_data.t_sys_us = t_new_us
    imu_data.transform_to_world(qs=cs_gt.qs)

    assert len(t_new_us) == len(imu_data) == len(cs_gt), (
        f"Length mismatch, {len(t_new_us)} != {len(cs_gt)} != {len(imu_data)}"
    )
    print(f"Data loaded successfully, {len(cs_gt)}")


# points = [1, 2, 3, 4]
# paths = []
# path = []

# for p in points:
#     path.append(p)
#     paths.append(path.copy())

# print(paths)

# points = [1, 2, 3, 4]
# paths = list(accumulate(points, lambda acc, x: acc + [x], initial=[]))[1:]
# print(paths)


# p = np.array([1, 2, 3])

# x90 = Quaternion(axis=[1, 0, 0], angle=-np.pi/2)

# q_ = x90
# print(q_.rotation_matrix.T)

# fp = FilePath("./dataset", PERSON_LIST)
# flatten = fp.flatten()
# flatten0 = flatten[0]
# # print(flatten0)

#

# a0 = imu_data.acce[0]
# q0 = imu_data.unit_ahrs[0]

# NOTE: AHRS旋转加速度 和 Cam 旋转加速度不一致的原因是 AHRS 和 CAM 的参考坐标系是不一致的。
# print(a0, "->", q0.rotation_matrix @ a0)

# ahrs = imu_data.unit_ahrs
# ahrs_Rs = np.array([it.rotation_matrix for it in ahrs])
# print(ahrs_Rs.shape)  # R_WI (N, 3, 3)
# print(imu_data.acce.shape)
# print(imu_data.gyro.shape)

# acc_W = np.einsum('ijk,ik->ij', ahrs_Rs, imu_data.acce)
# gyr_W = np.einsum('ijk,ik->ij', ahrs_Rs, imu_data.gyro)
# # print(acc_W, gyr_W)


# path = "dataset/20251111_174911_SM-G9900/251111-174954.db"
# rtab_data = RTABData(path)
# rtab_data.draw()


# 5965848223,0.08124501,-0.03428479,-0.01206458,0.07657032,9.14776134,3.20638227,0.67116857,0.46211508,-0.33180588,-0.47527596,1762860382299000,-39.46875000,-8.10000038,-23.53125000
# 5965801399,,0.98169613,-0.18979433,-0.00275505,0.01559882,0.01139609,0.08211301,-0.02406364,-0.70779383,0.13554430,0.12952611,0.68108636,1762860382298000

# a = np.array([0.07657032, 9.14776134, 3.20638227])
# q = [0.98169613, -0.18979433, -0.00275505, 0.01559882]

# q_base = Quaternion(axis=[1, 0, 0], angle=np.pi/2)


# q = Quaternion(q)
# print(q_base.rotation_matrix @ q.rotation_matrix @ a)
