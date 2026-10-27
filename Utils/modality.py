"""Modality masking for multi-sensor data batches allowing us to select only
certain modalities for the network.

``apply_modality_mask`` restricts a batch to the sensor modalities set in the
training config (via ``modality.use_aux_imu`` and ``modality.use_joints``).
"""

import dataclasses
from typing import Union

from Interface.data import IMUData_w_GT, IMUJointsData_w_GT


def apply_modality_mask(
    data: Union[IMUData_w_GT, IMUJointsData_w_GT],
    *,
    use_aux_imu: bool,
    use_joints: bool,
) -> Union[IMUData_w_GT, IMUJointsData_w_GT]:
    """Return a copy of data enforcing modality flags by dropping unused modality fields.

    Params
        data:
            A collated batch of type ``IMUData_w_GT`` or ``IMUJointsData_w_GT``.
        use_aux_imu:
            When False, slice all per-IMU fields (raw + corrected acc/gyro,
            time_ns, covariances) to ``[:, :, 0:1, ...]`` (main body only).
            ``corrected_acc`` / ``corrected_gyro`` are ``None`` whenever no
            correction stage ran, and are then passed through as ``None``.
        use_joints:
            When False and data has Joints (is a ``IMUJointsData_w_GT``), downcast to
            ``IMUData_w_GT``, dropping the joint fields.

    Returns
        The same object unchanged when both flags are *True* — no copy is made.
        Otherwise a new dataclass instance of the appropriate type.
    """
    if use_aux_imu and use_joints:
        return data

    # Enforce main IMU only
    if not use_aux_imu:
        imu_overrides: dict = dict(
            acc            = data.acc           [:, :, 0:1, :],
            gyro           = data.gyro          [:, :, 0:1, :],
            time_ns        = data.time_ns       [:, :, 0:1, :],
            acc_cov        = data.acc_cov       [:, :, 0:1, :, :],
            gyro_cov       = data.gyro_cov      [:, :, 0:1, :, :],
            # None (no correction stage ran) stays None — there is nothing to slice.
            corrected_acc  = None if data.corrected_acc  is None else data.corrected_acc [:, :, 0:1, :],
            corrected_gyro = None if data.corrected_gyro is None else data.corrected_gyro[:, :, 0:1, :],
        )
        data = dataclasses.replace(data, **imu_overrides)

    # Enforce joint fields drop
    if not use_joints and isinstance(data, IMUJointsData_w_GT):
        # Collect only the fields declared on IMUData_w_GT (and its parents
        # IMUData + GTState); this drops position/velocity/effort/joint_time_ns.
        base_field_names = {f.name for f in dataclasses.fields(IMUData_w_GT)}
        base_kwargs = {
            f.name: getattr(data, f.name)
            for f in dataclasses.fields(data)
            if f.name in base_field_names
        }
        data = IMUData_w_GT(**base_kwargs)

    return data
