from base import ARCoreData, IMUData, RTABData, UnitData

up = "./dataset/20251020_173007"
up = "/Users/qi/Resources/Dataset/Ruijie/80GT/2025-728-093141"
ud = UnitData(up)
gt_data = RTABData(ud.gt_path)
t_base_us = gt_data.t_sys_us[0]
print(t_base_us)
imu_data = IMUData(ud.imu_path, t_base_us=t_base_us)
cam_data = ARCoreData(ud.cam_path, z_up=True, t_base_us=t_base_us)

imu_data.save_csv("./imu_data.csv")
cam_data.save_csv("./cam_data.csv")
