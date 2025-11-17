from pathlib import Path

from .arcore_data import ARCoreData  # noqa
from .datatype import *  # noqa
from .imu_data import IMUData  # noqa
from .rtab_data import RTABData  # noqa


class FilePath:
    root_dir: Path
    persons: list[PersonData]

    def __init__(
        self,
        root_dir: str | Path,
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
