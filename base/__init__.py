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


class Dataset:
    root_dir: Path
    persons: list[PersonData]

    def __init__(self, root_dir: str | Path, person_ids: list[str] | None = None):
        self.root_dir = Path(root_dir)
        person_ids = person_ids if person_ids else self._load_dir_list()
        self.persons = [PersonData(self.root_dir.joinpath(pid)) for pid in person_ids]

    def _load_dir_list(self) -> list[str]:
        return [it.name for it in self.root_dir.iterdir() if it.is_dir()]

    def flatten(self) -> list[FlattenUnitData]:
        res = []
        for person in self.persons:
            for group in person.groups:
                for unit in group.data:
                    res.append(FlattenUnitData(person, group, unit))
        return res
