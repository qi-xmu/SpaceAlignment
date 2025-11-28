import rerun as rr
import rerun.blueprint as rrb

from base.datatype import ARCoreData, CalibrationData, IMUData, TimePoseSeries

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
                        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-20),
                        end=rrb.TimeRangeBoundary.cursor_relative(seconds=20),
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
                        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-20),
                        end=rrb.TimeRangeBoundary.cursor_relative(seconds=20),
                    )
                ),
            ),
            # visible=False,
        ),
        rrb.Spatial3DView(
            origin="/",
            name="World position",
            spatial_information=rrb.SpatialInformation(show_axes=True),
            eye_controls=rrb.EyeControls3D(
                kind="Orbital",
                tracking_entity="/world/groundtruth/sensor",
            ),
        ),
        column_shares=[0.40, 0.60],
    )
    rr.send_blueprint(blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def send_imu_cam_data(imu_data: IMUData, cam_data: ARCoreData | None = None):
    ts_imu = rr.TimeColumn("timestamp", timestamp=imu_data.t_sys_us * 1e-6)
    rr.send_columns(
        "/gyroscope",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.world_gyro),
    )
    rr.send_columns(
        "/accelerometer",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.world_acce),
    )

    qs = imu_data.ahrs_rots.as_quat()
    rr.send_columns(
        "/world/ahrs",
        indexes=[ts_imu],
        columns=rr.Transform3D.columns(quaternion=qs),
    )

    if cam_data:
        ts_cam = rr.TimeColumn("timestamp", timestamp=cam_data.t_sys_us * 1e-6)
        log_coordinate("/world/sensor", length=0.1, labels=["Sensor"], show_labels=True)
        qs = cam_data.sensor_rots.as_quat()
        rr.send_columns(
            "/world/sensor",
            indexes=[ts_cam],
            columns=rr.Transform3D.columns(
                quaternion=qs, translation=cam_data.sensor_ps
            ),
        )

        send_columns_path(
            "/world/sensor_path",
            indexes=[ts_cam],
            ps=cam_data.sensor_ps,
            labels=["Sensor"],
        )
    else:
        pass


def send_pose_data(ts: TimePoseSeries, cd: CalibrationData):
    times = rr.TimeColumn("timestamp", timestamp=ts.t_us * 1e-6)
    ts.transform_global(cd.tf_global)

    log_coordinate("/world/ahrs", length=0.1, show_labels=True, labels=["AHRS"])
    qs = ts.rots.as_quat()
    rr.send_columns(
        "/world/groundtruth",
        indexes=[times],
        columns=rr.Transform3D.columns(quaternion=qs, translation=ts.trans),
    )
    rr.send_columns(
        "/world/ahrs",
        indexes=[times],
        columns=rr.Transform3D.columns(translation=ts.trans),
    )

    send_columns_path(
        "/world/gt_path",
        indexes=[times],
        ps=ts.trans,
        static=True,
        labels=["Groundtruth"],
        colors=[[192, 72, 72]],
    )

    log_coordinate(
        "/world/groundtruth/sensor",
        length=0.1,
        labels=["Estimate"],
        show_labels=True,
    )

    tf_gs_local = cd.tf_local.inverse()
    rr.log(
        "/world/groundtruth/sensor",
        rr.Transform3D(
            mat3x3=tf_gs_local.rot.as_matrix(), translation=tf_gs_local.tran
        ),
        static=True,
    )
