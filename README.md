# 使用说明

每一次支架发生变化后，需要重新标定。

## 时间对齐算法

使用 `TimeMatch.py` 对齐时间

> 算法简介：匹配两个时间序列最近的时间点，阈值为高频率序列最大间隔的 1/2。

```python
if __name__ == "__main__":
    dataset_id = "20251020_173007"
    params = FilePaths(dataset_id)
    rtab_data = RTABData(params.rtab_db)
    arcore_data = ARCoreData(params.arcore_csv)

    # 获取时间轴数据
    ts1 = arcore_data.t_sys_us
    ts2 = rtab_data.node_t_us
    matches = match(ts1, ts2, dataset_id)
    match_draw(ts1, ts2, matches)
```

匹配完成之后，可以获得变量 `matches`，其中包含了匹配的结果，可以使用 `matches.draw()` 绘制匹配结果。

## 空间对齐算法 Hand Eye Calibration

使用 `Calibration.py` 标定匹配的 位姿对

> 算法简介：使用 Hand-Eye 标定算法，求解 相机到 标定基座的 位姿。默认使用 Park 算法。具体算法说明：https://flowus.cn/qiml/share/55c35643-f1de-4592-a1ea-f8d0d227cedd?code=1ZSBNL

```python
# 根据 索引对，转换为匹配的姿态对 Rs ts,
Rs_w1c, ts_w1c = arcore.matches_sensor_Rsts(matches, index=0)
# 注意， ts2 需要 inverse，根据 eye-in-hand 标定公式
Rs_nw2, ts_nw2 = rtab.matches_node_Rsts(matches, index=1, inverse=True)

# 标定结果，保存到文件
R_cn, t_cn = calibrate_Rgc(Rs_w1c, ts_w1c, Rs_nw2, ts_nw2)
evaluate_save(dataset_id, R_cn, t_cn, Rs_w1c, ts_w1c, Rs_nw2, ts_nw2)
```

标定时注意序列的先后顺序。假设先 A 后 B，则求解结果为 R_AB, t_AB。
其中，ts1 序列注意需要求逆。

## 其他

1. `dataset.py` 包含了数据集的路径，以及数据集的读取。

   ```python
   from dataset import FilePaths, ARCoreData, RTABData

   dataset_id = "20251020_173007"
   paths = FilePaths(dataset_id)
   arcore = ARCoreData(paths.arcore_csv, id=dataset_id)
   rtab = RTABData(paths.rtab_db)
   ```

2. `HandEye.py` 包含了 Hand-Eye 标定算法的实现。标定结果的评估和保存。
3. `TestOnlyModel.py` 使用 AHRS 旋转数据后，仅使用模型进行轨迹预测。
4. `CalibrationResults` 标定结果文件夹。
