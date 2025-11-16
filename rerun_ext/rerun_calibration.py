import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from base import ARCoreData, CalibrationData, IMUData, RTABData
from base.interpolate import get_time_series, pose_interpolate
from base.space import transform_local, transform_world

from . import log_coordinate, send_columns_path


def rerun_init(name: str):
    XYZ_AXIS_NAMES = ["x", "y", "z"]
    XYZ_AXIS_COLORS = [[(231, 76, 60), (39, 174, 96), (52, 120, 219)]]
    rr.init(name, spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.TimeSeriesView(
                origin="gyroscope",
                name="Gyroscope",
                overrides={
                    # type: ignore[arg-type]
                    "/gyroscope": rr.SeriesLines.from_fields(
                        names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                    ),
                },
                axis_x=rrb.archetypes.TimeAxis(
                    view_range=rrb.TimeRange(
                        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-2),
                        end=rrb.TimeRangeBoundary.cursor_relative(seconds=2),
                    )
                ),
            ),
            rrb.TimeSeriesView(
                origin="accelerometer",
                name="Accelerometer",
                overrides={
                    # type: ignore[arg-type]
                    "/accelerometer": rr.SeriesLines.from_fields(
                        names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                    ),
                },
                axis_x=rrb.archetypes.TimeAxis(
                    view_range=rrb.TimeRange(
                        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-2),
                        end=rrb.TimeRangeBoundary.cursor_relative(seconds=2),
                    )
                ),
            ),
            visible=False,
        ),
        rrb.Spatial3DView(
            origin="/",
            name="World position",
            spatial_information=rrb.SpatialInformation(show_axes=True),
            eye_controls=rrb.EyeControls3D(
                kind="Orbital",
                tracking_entity="/world/W_TO_GT/groundtruth",
            ),
        ),
        column_shares=[0.40, 0.60],
    )
    rr.send_blueprint(blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def send_imu_cam_data(
    imu_data: IMUData,
    cam_data: ARCoreData | None = None,
    cd_ic: CalibrationData = CalibrationData(),
):
    ts_imu = rr.TimeColumn("timestamp", timestamp=imu_data.t_sys_us * 1e-6)
    rr.send_columns(
        "/gyroscope",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.gyro),
    )
    rr.send_columns(
        "/accelerometer",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.acce),
    )
    rr.log(
        "/world/W_TO_CAM",
        rr.Transform3D(
            mat3x3=cd_ic.rot_ref_sensor_gt, translation=cd_ic.tr_ref_sensor_gt
        ),
        static=True,
    )

    if cam_data:
        ts_cam = rr.TimeColumn("timestamp", timestamp=cam_data.system_t_us * 1e-6)
        log_coordinate(
            "/world/W_TO_CAM/sensor", length=0.1, labels=["Sensor"], show_labels=True
        )

        qs = [[q.x, q.y, q.z, q.w] for q in cam_data.sensor_qs]
        rr.send_columns(
            "/world/W_TO_CAM/sensor",
            indexes=[ts_cam],
            columns=rr.Transform3D.columns(
                quaternion=qs, translation=cam_data.sensor_ps
            ),
        )

        send_columns_path(
            "/world/W_TO_CAM/sensor_path",
            indexes=[ts_cam],
            ps=cam_data.sensor_ps,
            labels=["Sensor"],
        )


def send_gt_data(gt_data: RTABData, calibr_data: CalibrationData):
    gt_data.node_qs, gt_data.node_ps = transform_world(
        tf_world=(calibr_data.rot_ref_sensor_gt, calibr_data.tr_ref_sensor_gt),
        qs=gt_data.node_qs,
        ps=gt_data.node_ps,
    )
    # gt_data.get_calibr_series()
    #
    t_new_us = get_time_series([gt_data.node_t_us])
    cs = pose_interpolate(
        cs=gt_data.get_time_pose_series(),
        t_new_us=t_new_us,
    )
    gt_data.node_t_us, gt_data.node_qs, gt_data.node_ps = cs.ts_us, cs.qs, cs.ps

    times = rr.TimeColumn("timestamp", timestamp=gt_data.node_t_us * 1e-6)
    # rr.log(
    #     "/world/W_TO_GT",
    #     rr.Transform3D(
    #         mat3x3=calibr_data.rot_ref_sensor_gt,
    #         translation=calibr_data.tr_ref_sensor_gt,
    #     ),
    #     static=True,
    # )

    log_coordinate(
        "/world/W_TO_GT/groundtruth",
        length=0.1,
        labels=["Groundtruth"],
        show_labels=True,
    )
    qs = np.array([[q.x, q.y, q.z, q.w] for q in gt_data.node_qs])
    rr.send_columns(
        "/world/W_TO_GT/groundtruth",
        indexes=[times],
        columns=rr.Transform3D.columns(quaternion=qs, translation=gt_data.node_ps),
    )

    send_columns_path(
        "/world/W_TO_GT/gt_path",
        indexes=[times],
        ps=gt_data.node_ps,
        static=True,
        labels=["Groundtruth"],
        colors=[[192, 72, 72]],
    )

    log_coordinate(
        "/world/W_TO_GT/groundtruth/sensor",
        length=0.1,
        labels=["Estimate"],
        show_labels=True,
    )
    rr.log(
        "/world/W_TO_GT/groundtruth/sensor",
        rr.Transform3D(
            mat3x3=calibr_data.rot_gt_sensor,
            translation=calibr_data.tr_gt_sensor,
        ),
        static=True,
    )
