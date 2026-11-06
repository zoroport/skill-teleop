"""Passive ROS2 collector plus local D435 reader; never commands a robot."""
import argparse
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from skill_teleop_interfaces.msg import ArmCommand, ArmState, HandCommand, HandState

from .camera import CameraProducer
from .schema import clock_identity, message_record
from .storage import EpisodeWriter


class Collector(Node):
    def __init__(self, args):
        super().__init__("skill_episode_collector")
        self.args, self.writer = args, None
        self.lock = threading.RLock()
        self.latest = {}
        self.event_seq = 0
        self.last_report = 0
        for name, cls, topic in (
            ("arm_state", ArmState, f"/{args.side}/collection/arm_state"),
            ("arm_command", ArmCommand, f"/{args.side}/collection/arm_command"),
            ("hand_state", HandState, "/xhand/collection/state"),
            ("hand_command", HandCommand, "/xhand/collection/command"),
        ):
            self.create_subscription(cls, topic, lambda m, n=name: self.receive(message_record(n, m)), 200)
        for name in ("start", "success", "failed", "mark_approach", "mark_grasp", "mark_screw", "mark_release"):
            self.create_service(Trigger, f"/collection/{name}",
                                lambda req, res, n=name: self.handle_service(n, res))
        self.camera = CameraProducer(self.receive, serial=args.camera_serial)
        self.camera.start()
        self.timer = self.create_timer(.5, self.monitor)

    def receive(self, record):
        with self.lock:
            if record["stream"] == "arm_command" and record.get("kind") == "input":
                record["stream"] = "arm_input"
            self.latest[record["stream"]] = record
            if self.writer:
                self.writer.submit(record)

    def readiness(self):
        now, age = time.monotonic_ns(), round(self.args.max_age_ms * 1e6)
        for stream in ("arm_state", "hand_state", "camera", "arm_command", "hand_command"):
            sample = self.latest.get(stream)
            if sample is None or not sample["valid"]:
                return f"Waiting for valid {stream}: {(sample or {}).get('error', 'no messages')}"
            if stream.endswith("state") or stream == "camera":
                if not 0 <= now - sample["t_ns"] <= age:
                    return f"Stale {stream}"
        for stream in ("arm_state", "hand_state"):
            if self.latest[stream]["metadata"].get("clock") != clock_identity():
                return f"{stream} uses another host/boot; this collector requires one Linux host"
            command = self.latest[stream.replace("state", "command")]
            if command["session"] != self.latest[stream]["session"]:
                return f"{stream} command belongs to another session"
        if not self.args.allow_simulated and not self.latest["hand_state"]["metadata"].get("hardware", False):
            return "Simulated hand; use --allow-simulated only for offline integration tests"
        return None

    def event(self, stage):
        record = {"stream": "event", "t_ns": time.monotonic_ns(), "seq": self.event_seq,
                  "valid": True, "stage": stage}
        self.event_seq += 1
        if self.writer:
            self.writer.submit(record)

    def finish(self, status):
        writer, self.writer = self.writer, None
        if writer:
            path = writer.finish(status)
            self.get_logger().info(f"Episode {writer.metadata['status']}: {path}")
            return f"{writer.metadata['status']}: {path}"
        raise ValueError("No episode is recording")

    def handle_service(self, action, response):
        with self.lock:
            try:
                if action == "start":
                    if self.writer:
                        raise ValueError("Already recording")
                    problem = self.readiness()
                    if problem:
                        raise ValueError(problem)
                    config = {"side": self.args.side, "nominal_hz": 20,
                              "max_age_ms": self.args.max_age_ms,
                              "arm": self.latest["arm_state"]["metadata"],
                              "hand": self.latest["hand_state"]["metadata"],
                              "camera": self.latest["camera"]["metadata"]}
                    self.writer = EpisodeWriter(self.args.data_dir, config)
                    # Seed observations and held commands at episode start with original timestamps.
                    for name in ("arm_state", "arm_command", "hand_state", "hand_command", "camera"):
                        self.writer.submit(self.latest[name])
                    self.event("approach")
                    response.message = str(self.writer.path)
                elif action in ("success", "failed"):
                    self.event(action)
                    response.message = self.finish(action)
                else:
                    if not self.writer:
                        raise ValueError("Start recording before marking stages")
                    self.event(action.removeprefix("mark_"))
                    response.message = action
                response.success = True
            except Exception as exc:
                response.success, response.message = False, str(exc)
        return response

    def monitor(self):
        with self.lock:
            if self.writer and self.writer.error:
                self.get_logger().error(self.writer.error)
                self.finish("failed")
            problem = self.readiness()
            if problem and time.monotonic() - self.last_report > 5:
                self.get_logger().warning(problem)
                self.last_report = time.monotonic()

    def destroy_node(self):
        self.camera.close()
        with self.lock:
            if self.writer:
                self.finish("interrupted")
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--data-dir", type=Path, default=Path("data/rm_xhand_raw"))
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--max-age-ms", type=float, default=100)
    parser.add_argument("--allow-simulated", action="store_true")
    args, ros_args = parser.parse_known_args()
    if not 0 < args.max_age_ms < float("inf"):
        parser.error("max-age-ms must be positive and finite")
    rclpy.init(args=ros_args)
    node = None
    try:
        node = Collector(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
