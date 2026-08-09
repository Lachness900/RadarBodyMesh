"""Reviewed SMPL joint presets for the mmYoga reference-pose viewer."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


POSE_LABELS = (
    "standing_pose",
    "t_pose",
    "squat",
    "angle_pose",
)

# SMPL's 24-joint order. Joint 0 is the pelvis/global orientation; the
# remaining 23 rotations form body_pose in the official smplx implementation.
JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
}


def _euler_xyz_to_axis_angle(
    x_degrees: float = 0,
    y_degrees: float = 0,
    z_degrees: float = 0,
) -> NDArray[np.float64]:
    """Convert local XYZ Euler degrees to one SMPL axis-angle vector."""

    x, y, z = np.radians([x_degrees, y_degrees, z_degrees])
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rotation = rotation_z @ rotation_y @ rotation_x

    cosine = float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2 * math.sin(angle)
    return axis * angle


def _set_joint(
    pose: NDArray[np.float64],
    joint: str,
    *,
    x: float = 0,
    y: float = 0,
    z: float = 0,
) -> None:
    """Set one local joint rotation using readable degree values."""

    pose[JOINTS[joint]] = _euler_xyz_to_axis_angle(x, y, z)


def _standing_pose() -> NDArray[np.float64]:
    """Natural upright stance with straight arms close to the torso."""

    pose = np.zeros((24, 3), dtype=np.float64)
    _set_joint(pose, "left_shoulder", z=-86)
    _set_joint(pose, "right_shoulder", z=86)
    return pose


def _squat_pose() -> NDArray[np.float64]:
    """Shoulder-width squat with vertical shins and a forward torso hinge."""

    pose = np.zeros((24, 3), dtype=np.float64)

    # Tilt the pelvis with the torso, then counter-rotate both hips so the
    # thighs lower toward horizontal while the shins stay nearly vertical.
    _set_joint(pose, "pelvis", x=15)
    _set_joint(pose, "left_hip", x=-119.25, y=17, z=5)
    _set_joint(pose, "right_hip", x=-123, y=-17, z=-5)
    _set_joint(pose, "left_knee", x=104.25, z=40)
    _set_joint(pose, "right_knee", x=108, z=-40)

    # Point both feet straight ahead. The small left-leg asymmetry compensates
    # for the source SMPL mesh so both complete soles share one ground plane.
    _set_joint(pose, "left_ankle", y=-60)
    _set_joint(pose, "right_ankle", y=60)

    # Distribute the remaining forward lean across the spine instead of one
    # sharp waist bend, which avoids the inflated abdomen seen previously.
    _set_joint(pose, "spine1", x=10)
    _set_joint(pose, "spine2", x=6)
    _set_joint(pose, "spine3", x=3)
    _set_joint(pose, "neck", x=-9)

    # Rotate both straight arms forward in parallel. Elbows and wrists remain
    # neutral so the hands stay separate and retain their original shape.
    _set_joint(pose, "left_shoulder", y=-90)
    _set_joint(pose, "right_shoulder", y=90)
    return pose


def _angle_pose() -> NDArray[np.float64]:
    """Konasana-style side bend with straight legs and one overhead arm."""

    pose = np.zeros((24, 3), dtype=np.float64)

    # Keep both knees straight and spread the feet evenly. Opposite ankle
    # rotations return the soles to the floor after the legs are abducted.
    _set_joint(pose, "left_hip", z=15)
    _set_joint(pose, "right_hip", z=-15)
    _set_joint(pose, "left_ankle", z=-15)
    _set_joint(pose, "right_ankle", z=15)

    # Build one smooth lateral curve through the three spine joints. The neck
    # and head partially counter-rotate so the face does not tip excessively.
    _set_joint(pose, "spine1", z=-17)
    _set_joint(pose, "spine2", z=-18)
    _set_joint(pose, "spine3", z=-20)
    _set_joint(pose, "neck", z=8)
    _set_joint(pose, "head", z=6)

    # The lower-side arm follows the outside leg. The upper-side arm crosses
    # above the head and continues the direction of the lateral stretch.
    _set_joint(pose, "left_shoulder", x=1, y=-8, z=-43)
    _set_joint(pose, "right_shoulder", x=60, z=-88)
    _set_joint(pose, "right_elbow", x=60)
    _set_joint(pose, "right_wrist", x=62, y=-40, z=11)
    return pose


def get_pose_axis_angles(pose_label: str) -> NDArray[np.float64]:
    """Return all 24 SMPL local rotations for one approved project label."""

    presets = {
        "standing_pose": _standing_pose,
        "t_pose": lambda: np.zeros((24, 3), dtype=np.float64),
        "squat": _squat_pose,
        "angle_pose": _angle_pose,
    }
    try:
        return presets[pose_label]()
    except KeyError as error:
        raise ValueError(f"No SMPL reference preset for {pose_label}") from error


def get_pose_translation(pose_label: str) -> NDArray[np.float64]:
    """Return the global root translation needed for one reference pose.

    SMPL keeps the pelvis at the template origin, so poses that lower the body
    (like a squat) would otherwise lift the feet off the floor. The squat
    translation returns the feet to the same ground plane as the T pose.
    """

    if pose_label == "squat":
        return np.array([0.0, -0.355, 0.0], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)
