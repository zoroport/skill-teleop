"""Home state machine tests with no ROS runtime or hardware access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "02_HandAndArm_control/Arm"))
from arm_home import ArmHome


class SDK:
    def __init__(self, dof):
        self.q = [10.] * dof
        self.kind, self.ret, self.stop_count = 0, 0, 0
        self.moves = []
    def rm_get_arm_current_trajectory(self):
        return dict(return_code=0, trajectory_type=self.kind, data=self.q.copy())
    def rm_get_joint_min_pos(self): return 0, [-180.] * len(self.q)
    def rm_get_joint_max_pos(self): return 0, [180.] * len(self.q)
    def rm_movej(self, *args): self.moves.append(args); return self.ret
    def rm_set_arm_slow_stop(self): self.stop_count += 1; return 0


class HomeTests(unittest.TestCase):
    def make(self, root, dof=6):
        events = []
        session = SimpleNamespace(is_teleoperating=False, motion_guard=None,
                                  _emit=lambda *a, **k: events.append((a, k)))
        controller = SimpleNamespace(name="left", ip="fake", robot_info={"arm_dof": dof},
            robot=SDK(dof), connection_generation=1, logger=SimpleNamespace(error=lambda _: None))
        return ArmHome(controller, session, Path(root)/"home.json"), controller, session, events

    def test_taught_home_6_7_dof_nonblocking_and_measured_arrival(self):
        for dof in (6, 7):
            with tempfile.TemporaryDirectory() as root:
                home, c, s, events = self.make(root, dof)
                with self.assertRaises(ValueError): home.start()
                home.capture()
                self.assertEqual(c.robot.moves, [])
                c.robot.q = [0.] * dof
                home.start()
                self.assertEqual(c.robot.moves, [([10.] * dof, 10, 0, 0, 0)])
                self.assertTrue(s.motion_guard)
                self.assertFalse(events[0][0][1])
                home.poll(); self.assertEqual(home.phase, "MOVING")
                c.robot.q = [10.] * dof
                for _ in range(3): home.poll()
                self.assertEqual(home.phase, "ARRIVED")
                self.assertIsNone(s.motion_guard)

    def test_active_teleop_limits_and_identity_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            h, c, s, _ = self.make(root)
            s.is_teleoperating = True
            with self.assertRaises(ValueError): h.capture()
            s.is_teleoperating = False
            h.capture()
            data = json.loads(h.path.read_text())
            for field, value in (("ip", "other"), ("arm_dof", 7), ("joints", [181.] * 6)):
                h.path.write_text(json.dumps(dict(data, **{field: value})))
                with self.assertRaises(ValueError): h.start()
            self.assertEqual(c.robot.moves, [])

    def test_failure_does_not_retry_and_explicit_stop_unlocks(self):
        with tempfile.TemporaryDirectory() as root:
            h, c, s, _ = self.make(root)
            h.capture(); c.robot.ret = -2
            with self.assertRaises(RuntimeError): h.start()
            self.assertEqual(len(c.robot.moves), 1)
            self.assertEqual(c.robot.stop_count, 1)
            self.assertEqual(h.phase, "FAULT")
            self.assertTrue(s.motion_guard)
            with self.assertRaises(ValueError): h.start()
            h.cancel(); h.poll()
            self.assertIsNone(s.motion_guard)

    def test_timeout_and_reconnect_stop_and_keep_guard(self):
        for reconnect in (False, True):
            with tempfile.TemporaryDirectory() as root:
                h, c, s, _ = self.make(root)
                h.capture(); h.start(); c.robot.q = [0.] * 6
                if reconnect: c.connection_generation += 1
                with patch("arm_home.time.monotonic", return_value=h.deadline + 1): h.poll()
                self.assertEqual(h.phase, "FAULT")
                self.assertEqual(c.robot.stop_count, 1)
                self.assertTrue(s.motion_guard)


if __name__ == "__main__":
    unittest.main()
