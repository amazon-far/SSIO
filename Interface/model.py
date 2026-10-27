"""Interface-side view of the network contract.

``INetwork`` is re-exported, not redefined: its canonical home is
``Model/Network.py``. This is the same class object, so
``instantiate(INetwork, ...)``'s isinstance guard is unaffected by the re-export.
"""

from Model.Network import INetwork    # noqa: F401  (re-export)

from .data import IMUData_w_GT, JointData


def get_est_velocity(model: INetwork, data: IMUData_w_GT) -> INetwork.Output:
    """Run the AirIO network with the data's attitude as conditioning.

    Builds ``INetwork.Input`` by Log-mapping the body→world SO3 rotation
    extracted from ``data.gt_poses`` to its so(3) tangent (the form the
    orientation encoder consumes), then forwards through the model. This is
    the AirIO contract: attitude is a known input. Plain loaders carry GT
    attitude; ``Sequence.WindowIntegratedOrientation`` swaps in gyro-integrated
    attitude — the network consumes whatever the data carries. Proprioceptive
    joints are passed through when the batch carries them (``JointData``); plain
    IMU batches pass ``joints=None``.

    Which IMU pair the network reads is the network's own ``imu_input`` setting
    (``Model/airio.py::select_imu``), resolved inside ``forward``; the Δv reference
    and PGO preintegration select their stream independently via
    ``train.dv_imu_input``.

    Attitude source: ``data.input_rot`` when an orientation wrapper populated it
    (gyro-integrated / AHRS / external estimator), else the GT rotation from
    ``data.gt_poses``. This is the ONLY consumer that follows ``input_rot`` — the
    pseudo-label Δv frame and PGO preintegration deliberately stay on GT.
    """
    rot = data.input_rot if data.input_rot is not None else data.gt_poses.rotation()
    rot_so3 = rot.Log()                              # SO3 → so3 [B, N, 3]
    joints  = data if isinstance(data, JointData) else None
    return model(INetwork.Input(imu=data, rot=rot_so3, joints=joints))
