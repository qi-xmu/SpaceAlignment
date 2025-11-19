from pathlib import Path
from typing import AnyStr, Callable, List, Tuple

from base.datatype import Dataset


def dataset_action(
    ds: Dataset, action: Callable, *args, **kwargs
) -> List[Tuple[Path, AnyStr]]:
    idx = 0
    results = []
    for p in ds.persons:
        for g in p.groups:
            for u in g.units:
                try:
                    idx += 1
                    print(f"\n{idx} {u.base_dir} ...")
                    action(u, *args, **kwargs)
                except Exception as e:
                    print(e)
                    results.append((u.base_dir, str(e)))
    return results
