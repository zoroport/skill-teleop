"""Shared on-disk contract. All pose quaternions are xyzw, joints are radians."""
import json
import os
import socket
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA_VERSION = 1
ACTION_SPEC = {
    "name": "rm_work_pose_rot6d_xhand_v1", "dim": 21,
    "position": [0, 3], "rotation_columns": [3, 9], "xhand": [9, 21],
    "rotation_order": "R[:,0] followed by R[:,1]", "units": "metres,radians",
    "hand_order": "SDK finger_command[0:12]",
}
STREAMS = ("arm_state", "arm_command", "hand_state", "hand_command", "camera", "event")


def clock_identity():
    return {"host": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def vector(value, size=None):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or (size is not None and a.size != size) or not np.isfinite(a).all():
        raise ValueError(f"Expected finite vector of size {size}, got {a.shape}")
    return a


def pose_to_action(pose, hand):
    pose, hand = vector(pose, 7), vector(hand, 12)
    if np.linalg.norm(pose[3:]) < 1e-8:
        raise ValueError("Zero quaternion")
    r = Rotation.from_quat(pose[3:]).as_matrix()
    return np.concatenate((pose[:3], r[:, 0], r[:, 1], hand)).astype(np.float32)


def action_to_pose(action):
    a = vector(action, 21)
    first = a[3:6]
    if np.linalg.norm(first) < 1e-8:
        raise ValueError("Degenerate rotation column")
    first = first / np.linalg.norm(first)
    second = a[6:9] - first * np.dot(first, a[6:9])
    if np.linalg.norm(second) < 1e-8:
        raise ValueError("Collinear rotation columns")
    second /= np.linalg.norm(second)
    rotation = np.column_stack((first, second, np.cross(first, second)))
    return np.r_[a[:3], Rotation.from_matrix(rotation).as_quat()], a[9:].copy()


def pose_array(pose):
    return [pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def message_record(stream, msg):
    """Convert generated ROS messages to portable dicts (no ROS objects in pickle)."""
    result = {"stream": stream, "t_ns": int(msg.monotonic_ns),
              "seq": int(msg.sequence), "session": int(msg.session),
              "valid": bool(msg.valid), "error": msg.error,
              "ros_ns": msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
              "frame_id": msg.header.frame_id}
    for name in ("active", "mode", "kind", "return_code", "call_start_ns", "call_end_ns",
                 "tactile_valid"):
        if hasattr(msg, name):
            result[name] = getattr(msg, name)
    for name in ("joint_names", "positions", "velocities", "sdk_pose", "tactile", "tactile_shape"):
        if hasattr(msg, name):
            result[name] = list(getattr(msg, name))
    for name in ("pose", "target", "raw_delta"):
        if hasattr(msg, name):
            result[name] = pose_array(getattr(msg, name))
    if hasattr(msg, "metadata_json"):
        result["metadata"] = json.loads(msg.metadata_json or "{}")
    return result
