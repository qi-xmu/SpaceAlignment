import os
from pathlib import Path
from .arcore_data import ARCoreData  # noqa
from .rtab_data import RTABData  # noqa
from .imu_data import IMUData  # noqa
from .datatype import PersonData, FlattenUnitData, CalibrationSeries, Pose, GroupData  # noqa


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
            map(lambda person_id: PersonData(
                self.root_dir, person_id), person_ids)
        )

    def flatten(self) -> list[FlattenUnitData]:
        res = []
        for person in self.persons:
            for group in person.groups:
                for unit in group.data:
                    res.append(FlattenUnitData(person, group, unit))
        return res


class UnitPath:
    ARCORE_CSV = "{}/cam.csv"
    IMU_CSV = "{}/imu.csv"

    imu_path: Path | str
    cam_path: Path | str    # ARCore
    gt_path: Path | str

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        _, _, device_name = self.base_dir.name.split("_")

        self.cam_path = self.ARCORE_CSV.format(base_dir)
        self.imu_path = self.IMU_CSV.format(base_dir)
        self.dataset_id = self.base_dir.parent
        self.device_name = device_name

        # 检查此文件夹下文件，选中第一个后缀为 db 的文件
        for file in os.listdir(base_dir):
            if file.endswith(".db"):
                self.gt_path = os.path.join(base_dir, file)
                break
        else:
            raise FileNotFoundError(f"No .db file found in {base_dir}")

    def target(self, target_name) -> Path:
        return self.base_dir.joinpath(target_name)

    def group(self, name: str) -> Path | None:
        strs = self.base_dir.as_posix().split("/Calibration")
        if len(strs) == 2:
            return Path(strs[0]).joinpath(name)
        else:
            return None
