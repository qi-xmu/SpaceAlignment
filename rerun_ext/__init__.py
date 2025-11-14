import rerun as rr
import numpy as np
from itertools import accumulate
#


def log_coordinate(
    entity_path: str,
    view_coordinate: rr.AsComponents = rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    length: float = 1.0,
    radii=None,
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=["X", "Y", "Z"],
    show_labels=None
):
    rr.log(
        entity_path,
        view_coordinate,
        rr.Arrows3D(
            vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * length,
            radii=radii,
            colors=colors,
            labels=labels,
            show_labels=show_labels
        ),
        static=True
    )


def send_columns_path(
    entity_path: str,
    indexes,
    ps: np.ndarray | list[np.ndarray],
    static=True,
    *args,
    **kwargs
):
    def _gen_path(ps: np.ndarray | list[np.ndarray]):
        paths = list(accumulate(ps, lambda acc, x: acc + [x], initial=[]))
        return paths[1:]

    if static is False:
        rr.send_columns(
            entity_path,
            indexes=indexes,
            columns=rr.LineStrips3D.columns(
                strips=_gen_path(ps),
                *args,
                **kwargs
            )
        )
    else:
        rr.log(
            entity_path,
            rr.LineStrips3D(strips=[ps], *args, **kwargs,),
            static=True
        )
