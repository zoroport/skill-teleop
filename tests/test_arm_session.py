"""No robot connection: exercise the existing pose control with a fake SDK wrapper."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose

ARM = Path(__file__).resolve().parents[3] / "02_HandAndArm_control/Arm"
sys.path.insert(0, str(ARM))
# The wrapper only needs these names at import time. No real SDK initialization.
stub = types.ModuleType("Robotic_Arm.rm_robot_interface")
stub.RoboticArm = object
stub.rm_api_version = lambda: "test"
stub.rm_thread_mode_e = int
sys.modules.setdefault("Robotic_Arm", types.ModuleType("Robotic_Arm"))
sys.modules["Robotic_Arm.rm_robot_interface"] = stub
from arm_teleop_session import ArmTeleopSession


class FakeController:
    connection_generation = 1
    def __init__(self):
        self.targets, self.ret, self.reconnect = [], 0, False
    def get_current_pose_matrix(self):
        return np.eye(4), None
    def quaternion_to_euler(self, q):
        return Rotation.from_quat(q).as_euler("xyz")
    def movep_canfd(self, target):
        self.targets.append(target)
        if self.reconnect:
            self.connection_generation += 1
        return self.ret
    def slow_stop(self):
        return 0


class ArmSessionTests(unittest.TestCase):
    def session(self, events=None):
        controller = FakeController()
        logger = types.SimpleNamespace(info=lambda x: None, warn=lambda x: None)
        session = ArmTeleopSession("left", controller, logger, True, .2, 1.5, True, 0, False,
                                  event_callback=events.append if events is not None else None)
        session.start()
        return controller, session

    def test_telemetry_does_not_change_targets(self):
        events = []
        c1, s1 = self.session()
        c2, s2 = self.session(events)
        msg = Pose(); msg.orientation.w = 1.; msg.position.x = .8
        for _ in range(4):
            s1.handle_delta_pose(msg); s2.handle_delta_pose(msg)
        np.testing.assert_allclose(c1.targets, c2.targets)
        self.assertAlmostEqual(c1.targets[0][0], .2)
        commands = [e for e in events if e["kind"] == "command"]
        self.assertEqual(len(commands), 4)
        self.assertTrue(all(e["valid"] for e in commands))
        self.assertTrue(all(e["call_start_ns"] <= e["call_end_ns"] for e in commands))

    def test_invalid_pose_failure_and_reconnect(self):
        events = []
        c, s = self.session(events)
        msg = Pose(); msg.orientation.w = 0.  # explicitly zero quaternion
        s.handle_delta_pose(msg)
        self.assertFalse(c.targets)
        self.assertEqual(events[-1]["kind"], "rejected")
        msg.orientation.w = 1.
        c.ret = -1; s.handle_delta_pose(msg)
        self.assertFalse(events[-1]["valid"])
        c.ret, c.reconnect = 0, True; s.handle_delta_pose(msg)
        self.assertFalse(events[-1]["valid"])
        s.stop()
        self.assertFalse(events[-1]["valid"])
        count = len(c.targets)
        s.handle_delta_pose(msg)
        self.assertEqual(len(c.targets), count)


if __name__ == "__main__":
    unittest.main()
