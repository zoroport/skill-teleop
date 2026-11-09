"""ROS2 Xhand nut skill. Hardware is enabled only with --hardware."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from skill_teleop_interfaces.msg import HandCommand, HandState

from .hand_device import HandDevice
from .schema import atomic_json, clock_identity, vector

RELEASE_CAPTURE_TOLERANCE_DEG = 3.0


class HandSkillNode(Node):
    def __init__(self, args):
        super().__init__("xhand_nut_skill")
        self.args = args
        spec = importlib.util.spec_from_file_location("nut_policy", args.policy_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.policy = module.NutPolicy(args.checkpoint)
        self.lower = module.MECHANICAL_LOWER[module.POLICY_TO_XHAND]
        self.upper = module.MECHANICAL_UPPER[module.POLICY_TO_XHAND]
        # The requested release/reset pose is the nut policy's initial target.
        self.release = self.validate_target(self.policy.initial_hardware)
        release_source = "nut_policy.initial_hardware"
        if args.release_file.exists():
            calibration = json.loads(args.release_file.read_text())
            if calibration.get("hand_id") != args.hand_id or calibration.get("order") != "hardware":
                raise ValueError("Release calibration must match hand_id and use hardware joint order")
            self.release = self.validate_target(calibration["positions"])
            release_source = str(args.release_file.resolve())
        self.device = HandDevice(args.serial_port, args.baud_rate, args.hand_id, args.hardware)
        self.state_pub = self.create_publisher(HandState, "/xhand/collection/state", 100)
        self.command_pub = self.create_publisher(HandCommand, "/xhand/collection/command", 100)
        self.seq_state = self.seq_command = 0
        self.session = time.monotonic_ns()
        self.mode, self.error, self.ready = "HOLD", "", False
        self.closed = False
        self.metadata = {"clock": clock_identity(), "hand_id": args.hand_id,
                         "serial_port": args.serial_port, "baud_rate": args.baud_rate,
                         "hardware": args.hardware, "policy_hz": args.hz,
                         "checkpoint": str(self.policy.checkpoint),
                         "checkpoint_sha256": hashlib.sha256(self.policy.checkpoint.read_bytes()).hexdigest(),
                         "joint_order": "SDK finger_state/finger_command[0:12]",
                         "release_positions": self.release.tolist(), "release_source": release_source,
                         "kp": 100, "ki": 0, "kd": 1, "tor_max": 100, "action_scale": .04167}
        self.transition = None
        try:
            self.device.open()
            self.measured, _ = self.device.read()
            self.target = self.measured.copy()
            self.send(self.target)  # Powered hold of measured pose, no automatic grasp.
        except BaseException:
            self.device.close()
            super().destroy_node()
            raise
        for name in ("grasp", "start", "stop", "release", "capture_release"):
            self.create_service(Trigger, f"/xhand/skill/{name}",
                                lambda request, response, action=name: self.handle_service(action, response))
        self.next_tick = time.monotonic()
        # A lightweight scheduler skips overdue slots instead of catching up in bursts.
        self.timer = self.create_timer(.002, self.tick)

    def validate_target(self, values, tolerance_rad=0.0):
        q = vector(values, 12)
        if not np.isfinite(tolerance_rad) or tolerance_rad < 0:
            raise ValueError("Invalid capture tolerance")
        # Tiny arithmetic allowance only for the inclusive capture-tolerance boundary.
        allowance = tolerance_rad + (1e-12 if tolerance_rad > 0 else 0)
        outside = np.flatnonzero((self.lower - q > allowance) | (q - self.upper > allowance))
        if outside.size:
            details = []
            for i in outside:
                excess = max(self.lower[i] - q[i], q[i] - self.upper[i])
                details.append(
                    f"joint[{i}]={q[i]:.9g} rad ({np.degrees(q[i]):.6g} deg), "
                    f"allowed=[{self.lower[i]:.9g}, {self.upper[i]:.9g}] rad, "
                    f"excess={excess:.9g} rad")
            raise ValueError("Hand target exceeds configured limits (SDK order, zero-based): "
                             + "; ".join(details)
                             + f"; allowed capture tolerance={np.degrees(tolerance_rad):g} deg")
        return np.clip(q, self.lower, self.upper)

    def header(self, msg, seq, t):
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.monotonic_ns, msg.sequence, msg.session = t, seq, self.session
        msg.mode = self.mode

    def handle_service(self, action, response):
        try:
            result_message = None
            if self.mode == "FAULT":
                raise RuntimeError(f"Restart node after resolving fault: {self.error}")
            if action == "start":
                if not self.ready or self.transition is not None:
                    raise ValueError("Complete grasp before starting the nut policy")
                self.mode = "SCREW"
            elif action == "stop":
                self.mode, self.transition = "HOLD", None
            elif action == "capture_release":
                if self.mode != "HOLD":
                    raise ValueError("Stop skill before capturing a measured release pose")
                measured = self.measured.copy()
                release = self.validate_target(measured, np.deg2rad(RELEASE_CAPTURE_TOLERANCE_DEG))
                clipped = np.flatnonzero(release != measured).tolist()
                self.args.release_file.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(self.args.release_file, {"positions": release.tolist(),
                            "order": "hardware", "hand_id": self.args.hand_id,
                            "measured_positions": measured.tolist(), "clipped_joint_indices": clipped,
                            "capture_tolerance_deg": RELEASE_CAPTURE_TOLERANCE_DEG})
                # Only adopt the new pose after saving succeeds. Feedback is never clipped.
                self.release = release
                self.metadata.update(release_positions=self.release.tolist(),
                                     release_source=str(self.args.release_file.resolve()))
                result_message = f"HOLD; release pose saved; capture tolerance={RELEASE_CAPTURE_TOLERANCE_DEG:g} deg"
                if clipped:
                    result_message += "; clipped to limits (SDK zero-based): " + "; ".join(
                        f"joint[{i}] {measured[i]:.9g} -> {release[i]:.9g} rad" for i in clipped)
            else:
                target = self.policy.initial_hardware if action == "grasp" else self.release
                if target is None:
                    raise ValueError("Configure or capture a release pose first")
                self.ready = False
                self.mode = "GRASP" if action == "grasp" else "RELEASE"
                self.transition = (time.monotonic(), self.target.copy(), self.validate_target(target))
            response.success, response.message = True, result_message or self.mode
        except Exception as exc:
            response.success, response.message = False, str(exc)
        return response

    def send(self, target):
        msg = HandCommand()
        msg.call_start_ns = time.monotonic_ns()
        msg.positions = np.asarray(target, dtype=float).tolist()
        try:
            self.device.send(target)
            msg.valid, msg.return_code = True, 0
            self.target = np.asarray(target).copy()
        except Exception as exc:
            msg.valid, msg.return_code, msg.error = False, getattr(exc, "return_code", -1), str(exc)
            raise
        finally:
            msg.call_end_ns = time.monotonic_ns()
            self.header(msg, self.seq_command, msg.call_end_ns)
            self.seq_command += 1
            self.command_pub.publish(msg)

    def tick(self):
        now = time.monotonic()
        if now < self.next_tick:
            return
        period = 1 / self.args.hz
        msg = HandState()
        try:
            if self.mode == "FAULT":
                raise RuntimeError(self.error)
            self.measured, tactile = self.device.read()
            observed = time.monotonic_ns()
            msg.positions = self.measured.tolist()
            msg.joint_names = [f"xhand_joint_{i}" for i in range(12)]
            msg.tactile_valid = tactile is not None
            if tactile is not None:
                msg.tactile, msg.tactile_shape = tactile.flatten().tolist(), list(tactile.shape)
            msg.valid = True
            self.header(msg, self.seq_state, observed)
            if self.mode == "SCREW":
                target = self.policy.step(self.measured)
            elif self.transition is not None:
                start, initial, final = self.transition
                fraction = min(1., (now - start) / 1.)
                target = initial + (final - initial) * fraction
                if fraction == 1 and np.max(np.abs(self.measured - final)) < .1:
                    self.ready = self.mode == "GRASP"
                    if self.ready:
                        self.policy.reset()
                    self.mode, self.transition = "HOLD", None
            else:
                target = self.target
            self.send(target)
        except Exception as exc:
            self.mode, self.error = "FAULT", str(exc)
            msg.valid, msg.error = False, self.error
            self.header(msg, self.seq_state, time.monotonic_ns())
        msg.metadata_json = json.dumps(self.metadata)
        self.seq_state += 1
        self.state_pub.publish(msg)
        self.next_tick += period
        end = time.monotonic()
        if self.next_tick <= end:
            self.next_tick += (int((end - self.next_tick) / period) + 1) * period

    def destroy_node(self):
        if not self.closed:
            self.closed = True
            self.device.close()  # Does not release, return home, or disable motors.
        super().destroy_node()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--baud-rate", type=int, default=3000000)
    parser.add_argument("--hand-id", type=int, default=22)
    parser.add_argument("--hz", type=float, default=20)
    parser.add_argument("--policy-module", type=Path, default=root / "dexscrew/xhand-deploy/nut_policy.py")
    parser.add_argument("--checkpoint", type=Path, default=root / "dexscrew/xhand-deploy/checkpoints/nutbolt_1019.pt")
    parser.add_argument("--release-file", type=Path, default=Path("calibration/xhand22_release.json"))
    args, ros_args = parser.parse_known_args()
    if not np.isfinite(args.hz) or not 0 < args.hz <= 20:
        parser.error("hz must be > 0 and <= 20")
    rclpy.init(args=ros_args)
    node = None
    try:
        node = HandSkillNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
