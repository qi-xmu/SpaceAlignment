from pathlib import Path

from .arcore_data import ARCoreData  # noqa
from .datatype import (  # noqa
    CalibrationData,
    FlattenUnitData,
    GroupData,
    PersonData,
    Pose,
    Poses,
    Time,
    TimePoseSeries,
    UnitData,
)
from .imu_data import IMUData  # noqa
from .rtab_data import RTABData  # noqa


class FilePath:
    root_dir: Path
    persons: list[PersonData]

    def __init__(
        self,
        root_dir: str | Path,
    ):
        self.root_dir = Path(root_dir)
        self.person_ids = self._load_dir_list()

        self.persons = list(
            map(lambda person_id: PersonData(self.root_dir, person_id), self.person_ids)
        )

    def _load_dir_list(self) -> list[str]:
        return [it.name for it in self.root_dir.iterdir() if it.is_dir()]

    def flatten(self) -> list[FlattenUnitData]:
        res = []
        for person in self.persons:
            for group in person.groups:
                for unit in group.data:
                    res.append(FlattenUnitData(person, group, unit))
        return res
