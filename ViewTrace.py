
from base import RTABData, UnitPath
import rerun as rr
import rerun.blueprint as rrb
from base.arcore_data import ARCoreData
from base.imu_data import IMUData
import numpy as np
from pyquaternion import Quaternion
import rerun_ext as rre


XYZ_AXIS_NAMES = ["x", "y", "z"]
XYZ_AXIS_COLORS = [[(231, 76, 60), (39, 174, 96), (52, 120, 219)]]

R = np.array(
    [
        [
            -0.17408725816304502,
            -0.9847204179934088,
            0.004396013211646107
        ],
        [
            0.9847153325259925,
            -0.1741074457651859,
            -0.004723475134465926
        ],
        [
            0.00541668104062114,
            0.0035065247763325642,
            0.999979181708547
        ]
    ],
    dtype=np.float64
)
t = np.array(
    [
        0.2508558581925533,
        -0.3113045250339728,
        -0.01401398149443091
    ],
    dtype=np.float64
)
q = Quaternion(matrix=R)

R1 = np.array([
    [
        0.07054862222116191,
        0.01722030678404854,
        0.9973596908522783
    ],
    [
        -0.033390754136349725,
        0.999331412843456,
        -0.014892442460899437
    ],
    [
        -0.9969493214004216,
        -0.032251950925534095,
        0.07107645334837709
    ]
],

)

t1 = np.array(
    [
        0.028815072028137448,
        0.003062417854157243,
        0.1278765283302863
    ],
)


def rerun_init():
    rr.init("IMU_View", spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.TimeSeriesView(
                origin="gyroscope",
                name="Gyroscope",
                overrides={
                    # type: ignore[arg-type]
                    "/gyroscope": rr.SeriesLines.from_fields(names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS),
                },
                axis_x=rrb.archetypes.TimeAxis(view_range=rrb.TimeRange(
                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-2),
                    end=rrb.TimeRangeBoundary.cursor_relative(seconds=2),
                ))
            ),
            rrb.TimeSeriesView(
                origin="accelerometer",
                name="Accelerometer",
                overrides={
                    # type: ignore[arg-type]
                    "/accelerometer": rr.SeriesLines.from_fields(names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS),
                },
                axis_x=rrb.archetypes.TimeAxis(view_range=rrb.TimeRange(
                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-2),
                    end=rrb.TimeRangeBoundary.cursor_relative(seconds=2),
                ))
            ),
            visible=False
        ),
        rrb.Spatial3DView(origin="/", name="World position",
                          spatial_information=rrb.SpatialInformation(
                              show_axes=True),
                          eye_controls=rrb.EyeControls3D(kind="Orbital", tracking_entity="/world/bracket/groundtruth")),
        column_shares=[0.45, 0.55],
    )
    rr.send_blueprint(blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def send_imu_data(imu_data: IMUData):
    times = rr.TimeColumn("timestamp", timestamp=imu_data.t_sys_us * 1e-6)
    rr.send_columns(
        "/gyroscope", indexes=[times], columns=rr.Scalars.columns(scalars=imu_data.gyro))
    rr.send_columns(
        "/accelerometer", indexes=[times], columns=rr.Scalars.columns(scalars=imu_data.acce))

    # qs = np.array([[q.x, q.y, q.z, q.w] for q in imu_data.unit_ahrs])
    # rr.log("/world/bracket/sensor/AHRS", rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    #        rr.Arrows3D(
    #            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    #            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    #            labels=["SensorX", "SensorY", "SensorZ"],
    #            show_labels=True
    #        ),  static=True)
    # rr.send_columns(
    #     "/world/bracket/sensor/pose", indexes=[times], columns=rr.Transform3D.columns(quaternion=qs)
    # )


def send_gt_data(gt_data: RTABData):
    times = rr.TimeColumn("timestamp", timestamp=gt_data.node_t_us * 1e-6)
    rr.log("/world/bracket", rr.Transform3D(mat3x3=R, translation=t), static=True)
    rre.log_coordinate(
        "/world/bracket/groundtruth",
        length=0.1,
        labels=["Groundtruth"],
        show_labels=True
    )

    qs = np.array([[q.x, q.y, q.z, q.w] for q in gt_data.node_qs])
    rr.send_columns(
        "/world/bracket/groundtruth",
        indexes=[times],
        columns=rr.Transform3D.columns(
            quaternion=qs,
            translation=gt_data.node_ps
        )
    )

    rre.send_columns_path(
        "/world/bracket/groundtruth_path",
        indexes=[times],
        ps=gt_data.node_ps,
        static=False
    )


def send_cam_data(cam_data: ARCoreData):
    times = rr.TimeColumn("timestamp", timestamp=cam_data.t_sys_us * 1e-6)

    rre.log_coordinate(
        "/world/sensor",
        length=0.1,
        labels=["Sensor"],
        show_labels=True
    )

    qs = np.array([[q.x, q.y, q.z, q.w] for q in cam_data.unit_qs_sensor])
    rr.send_columns(
        "/world/sensor",
        indexes=[times], columns=rr.Transform3D.columns(
            quaternion=qs,
            translation=cam_data.ps)
    )

    rre.send_columns_path(
        "/world/sensor_path",
        indexes=[times],
        ps=cam_data.ps,
        # static=False
    )

    rre.log_coordinate(
        "/world/sensor/estimate",
        length=0.1,
        labels=["Estimate"],
        show_labels=True
    )

    rr.log("/world/sensor/estimate",
           rr.Transform3D(mat3x3=R1, translation=t1), static=True)


path = "dataset/001/20251031_01_in/Calibration/20251031_095725_SM-G9900"
path = "dataset/001/20251031_01_in/20251031_101025_SM-G9900"
path = "dataset/001/20251031_01_in/20251031_102355_SM-G9900"
path = "dataset/001/20251031_01_in/20251031_103441_SM-G9900"
fp = UnitPath(path)

gt_data = RTABData(fp.gt_path)
imu_data = IMUData(fp.imu_path)
cam_data = ARCoreData(fp.cam_path)


print(q.axis, q.angle * 180 / np.pi)

rerun_init()
send_imu_data(imu_data)
send_cam_data(cam_data)
send_gt_data(gt_data)
