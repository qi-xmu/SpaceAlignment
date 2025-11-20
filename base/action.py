import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import AnyStr, Callable, List, Tuple

from base.datatype import Dataset, UnitData


def _inner_action(u: UnitData, action: Callable, *args, **kwargs):
    try:
        action(u, *args, **kwargs)
        return None
    except Exception as e:
        print(e)
        # 获取错误的堆栈信息
        err = traceback.format_exc()
        return (u.base_dir, str(err))


def dataset_action(
    ds: Dataset, action: Callable, *args, **kwargs
) -> List[Tuple[Path, AnyStr]]:
    idx = 0
    results = []
    units = ds.flatten()
    for u in units:
        idx += 1
        print(f"\n{idx} {u.base_dir} ...")
        res = _inner_action(u, action, *args, **kwargs)
        if res:
            results.append(res)

    return results


# 改进一个并行的版本
#
def dataset_action_pa(ds: Dataset, action: Callable, *args, **kwargs):
    idx = 0
    results = []
    units = ds.flatten()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(action, u, *args, **kwargs) for u in units]
        for future in as_completed(futures):
            idx += 1
            print(f"\n{idx} ...")
            res = future.result()
            if res:
                results.append(res)
    return results
