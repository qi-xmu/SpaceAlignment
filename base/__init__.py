from .calibrate import calibrate_unit
from .datatype import CalibrationData, UnitData


def load_calibration_data(
    *,
    unit: UnitData,
    using_rerun: bool = False,
):
    # 加载 校准数据
    try:
        cd = CalibrationData.from_json(unit.unit_calib_path)
    except Exception as _:
        print("-" * 20, f"标定 {unit.device_name}")
        unit.unit_calib_path = unit.base_dir / "CalibrationTemp.json"
        cd = calibrate_unit(unit, t_len_s=40, using_rerun=using_rerun)
    return cd
