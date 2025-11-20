from pathlib import Path
from typing import AnyStr, Callable, List, Tuple

from base.datatype import Dataset


def dataset_action(
    ds: Dataset, action: Callable, *args, **kwargs
) -> List[Tuple[Path, AnyStr]]:
    idx = 0
    results = []
    units = ds.flatten()
    for u in units:
        try:
            idx += 1
            print(f"\n{idx} {u.base_dir} ...")
            action(u, *args, **kwargs)
        except Exception as e:
            print(e)
            results.append((u.base_dir, str(e)))
    return results
