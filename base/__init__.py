import os
from pathlib import Path

from .arcore_data import ARCoreData  # noqa
from .datatype import (  # noqa
    CalibrationData,
    FlattenUnitData,
    GroupData,
    PersonData,
    Pose,
    Poses,
    TimePoseSeries,
)
from .imu_data import IMUData  # noqa
from .rtab_data import RTABData  # noqa

PERSON_LIST = ["001"]


class FilePath:
    root_dir: Path
    persons: list[PersonData]

    def __init__(
        self,
        root_dir: str,
        person_ids: list[str],
    ):
        self.root_dir = Path(root_dir)
        self.persons = list(
            map(lambda person_id: PersonData(self.root_dir, person_id), person_ids)
        )

    def flatten(self) -> list[FlattenUnitData]:
        res = []
        for person in self.persons:
            for group in person.groups:
                for unit in group.data:
                    res.append(FlattenUnitData(person, group, unit))
        return res


class UnitPath:
    _ARCORE_CSV = "{}/cam.csv"
    _IMU_CSV = "{}/imu.csv"
    _CALIBR_FILE = "Calibration_{}.json"

    imu_path: Path | str
    cam_path: Path | str  # ARCore
    gt_path: Path | str
    calibr_file: Path

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        _, _, device_name = self.base_dir.name.split("_")

        self.cam_path = self._ARCORE_CSV.format(base_dir)
        self.imu_path = self._IMU_CSV.format(base_dir)
        self.dataset_id = self.base_dir.parent
        self.device_name = device_name
        self.calibr_name = self._CALIBR_FILE.format(device_name)

        if "/Calibration" in str(self.base_dir):
            self.group_path = self.base_dir.parent.parent
        else:
            self.group_path = self.base_dir.parent

        # 检查此文件夹下文件，选中第一个后缀为 db 的文件
        for file in os.listdir(base_dir):
            if file.endswith(".db"):
                self.gt_path = os.path.join(base_dir, file)
                break
        else:
            raise FileNotFoundError(f"No .db file found in {base_dir}")

        calir_file = self.group_path.joinpath(self.calibr_name)
        if not calir_file.exists():
            calir_file = self.base_dir.joinpath("Calibration.json")
        self.calibr_file = calir_file

    def target(self, target_name) -> Path:
        return self.base_dir.joinpath(target_name)

    def __str__(self) -> str:
        return str(
            {
                "imu_path": self.imu_path,
                "cam_path": self.cam_path,
                "gt_path": self.gt_path,
                "calir_file": self.calibr_file,
            }
        )
