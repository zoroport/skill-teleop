"""Validate capture/release bounds without opening devices or starting ROS nodes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import json
import tempfile
from unittest.mock import patch
from types import MethodType

import numpy as np
from skill_teleop.hand_node import HandSkillNode


class HandLimits(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / "dexscrew/xhand-deploy/nut_policy.py"
        spec = importlib.util.spec_from_file_location("limits_policy", path)
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        self.bounds = SimpleNamespace(
            lower=policy.MECHANICAL_LOWER[policy.POLICY_TO_XHAND],
            upper=policy.MECHANICAL_UPPER[policy.POLICY_TO_XHAND])

    def test_sdk_order_and_boundary_acceptance(self):
        np.testing.assert_array_equal(self.bounds.lower, [0, -1.05, -.17, -.175] + [0] * 8)
        np.testing.assert_array_equal(self.bounds.upper, [1.83, 1.57, 1.83, .175] + [1.92] * 8)
        for value in (self.bounds.lower, self.bounds.upper):
            result = HandSkillNode.validate_target(self.bounds, value)
            np.testing.assert_array_equal(result, value)
            self.assertFalse(np.shares_memory(result, value))

    def test_error_reports_every_outside_joint_without_clipping(self):
        value = (self.bounds.lower + self.bounds.upper) / 2
        value[4], value[0] = -.001, 1.831
        original = value.copy()
        with self.assertRaises(ValueError) as caught:
            HandSkillNode.validate_target(self.bounds, value)
        message = str(caught.exception)
        for expected in ("SDK order, zero-based", "joint[0]=1.831 rad", "joint[4]=-0.001 rad",
                         "allowed=[0, 1.92] rad", "excess=0.001 rad"):
            self.assertIn(expected, message)
        np.testing.assert_array_equal(value, original)

    def test_capture_tolerance_inclusive_both_bounds(self):
        tolerance = np.deg2rad(3)
        for boundary, sign in ((self.bounds.lower, -1), (self.bounds.upper, 1)):
            measured = boundary + sign * tolerance
            np.testing.assert_array_equal(
                HandSkillNode.validate_target(self.bounds, measured, tolerance), boundary)
            with self.assertRaises(ValueError):
                HandSkillNode.validate_target(self.bounds, measured + sign * 1e-6, tolerance)
        for bad in (float('nan'), float('inf')):
            measured = self.bounds.lower.copy()
            measured[9] = bad
            with self.assertRaises(ValueError):
                HandSkillNode.validate_target(self.bounds, measured, tolerance)

    def test_capture_service_saves_clipped_pose_preserves_feedback_and_previous_on_failure(self):
        node = self.bounds
        node.validate_target = MethodType(HandSkillNode.validate_target, node)
        node.mode, node.metadata = "HOLD", {}
        node.measured = (node.lower + node.upper) / 2
        node.measured[9], node.measured[11] = -.0378143527, -.00290879654
        original = node.measured.copy()
        with tempfile.TemporaryDirectory() as temp:
            node.args = SimpleNamespace(release_file=Path(temp) / "release.json", hand_id=22)
            response = HandSkillNode.handle_service(node, "capture_release", SimpleNamespace())
            self.assertTrue(response.success, response.message)
            self.assertIn("joint[9]", response.message)
            saved = json.loads(node.args.release_file.read_text())
            self.assertEqual(saved["clipped_joint_indices"], [9, 11])
            self.assertEqual(saved["positions"][9], 0)
            self.assertEqual(saved["positions"][11], 0)
            np.testing.assert_array_equal(saved["measured_positions"], original)
            np.testing.assert_array_equal(node.measured, original)
            # Saved target passes the normal strict validation on reload/use.
            node.validate_target(saved["positions"])
            previous = node.release.copy()
            previous_file = node.args.release_file.read_bytes()
            node.measured[9] = -np.deg2rad(3.01)
            response = HandSkillNode.handle_service(node, "capture_release", SimpleNamespace())
            self.assertFalse(response.success)
            self.assertEqual(node.args.release_file.read_bytes(), previous_file)
            np.testing.assert_array_equal(node.release, previous)
            node.measured[9] = .1
            with patch("skill_teleop.hand_node.atomic_json", side_effect=OSError("disk full")):
                response = HandSkillNode.handle_service(node, "capture_release", SimpleNamespace())
            self.assertFalse(response.success)
            np.testing.assert_array_equal(node.release, previous)


if __name__ == "__main__":
    unittest.main()
