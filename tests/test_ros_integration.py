"""Real local ROS transport + fake RM/D435 + simulated Xhand, no device access.

Run separately from test_arm_session.py (which replaces the RM import with a stub).
Requires generated interfaces, torch, and the installed RM SDK for ctypes only.
"""
import json
import signal
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger, SetBool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "02_HandAndArm_control/Arm"))
# Only the SDK ctypes declarations are imported; FakeController owns all calls.
sys.path.append("/home/cc/miniforge3/envs/arm_hand118_py310/lib/python3.10/site-packages")
from Robotic_Arm.rm_robot_interface import rm_realtime_push_config_t, rm_realtime_arm_joint_state_t
from dual_arm_teleop_node import DualArmTeleopNode
from dual_arm_teleop_node import main as arm_main
from collection_telemetry import active_robot_errors
from skill_teleop.collect import Collector
from skill_teleop.hand_node import HandSkillNode
from skill_teleop.storage import read_episode


class FakeCamera:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def close(self): pass


class FakeSDK:
    def __init__(self):
        self.original = rm_realtime_push_config_t(enable=False, ip="127.0.0.1", port=8098).to_dict()
        self.settings = []
        self.errors = [0]  # Real SDK serializes this as {'err_len': 1, 'err': ['0']}.
        self.joints, self.trajectory_type, self.home_moves = [10.] * 6, 0, []
    def rm_get_current_work_frame(self): return 0, {"name": "base", "pose": [0]*6}
    def rm_get_current_tool_frame(self): return 0, {"name": "tool", "pose": [0]*6}
    def rm_get_realtime_push(self): return 0, self.original.copy()
    def rm_set_realtime_push(self, config): self.settings.append(config.to_dict()); return 0
    def rm_realtime_arm_state_call_back(self, cb): self.callback = cb
    def rm_get_arm_current_trajectory(self):
        return dict(return_code=0, trajectory_type=self.trajectory_type, data=self.joints.copy())
    def rm_get_joint_min_pos(self): return 0, [-180.] * 6
    def rm_get_joint_max_pos(self): return 0, [180.] * 6
    def rm_movej(self, *args): self.home_moves.append(args); self.trajectory_type = 1; return 0
    def rm_set_arm_slow_stop(self): self.trajectory_type = 0; return 0
    def push(self):
        state = rm_realtime_arm_joint_state_t()
        state.arm_ip = b"127.0.0.1"
        state.waypoint.position.x = .2
        state.waypoint.quaternion.w = 1.
        state.err.err_len = len(self.errors)
        for i, value in enumerate(self.errors): state.err.err[i] = value
        for i in range(6): state.joint_status.joint_position[i] = self.joints[i]
        self.callback(state)


class FakeController:
    def __init__(self, side):
        self.name, self.ip, self.port = side, "127.0.0.1", 8080
        self.thread_mode, self.connection_generation = 2, 1
        self.follow, self.trajectory_mode, self.radio = False, 0, 0
        self.robot_info = {"arm_dof": 6, "model": "SIMULATED"}
        self.robot = FakeSDK()
        self.commands = []
    def connect(self): pass
    def disconnect(self): pass
    def keepalive(self): pass
    def get_current_pose_matrix(self): return np.eye(4), None
    def quaternion_to_euler(self, q): return Rotation.from_quat(q).as_euler("xyz")
    def euler_to_quaternion(self, e): return Rotation.from_euler("xyz", e).as_quat()
    def movep_canfd(self, p): self.commands.append(p); return 0
    def slow_stop(self): return 0


class ROSIntegration(unittest.TestCase):
    def test_error_codes_zero_nonzero_and_malformed(self):
        for codes in ([], [0], ["0"], ["0x0", "000", 0]):
            self.assertEqual(active_robot_errors({"err_len": len(codes), "err": codes}), [])
        self.assertEqual(active_robot_errors({"err_len": 3, "err": ["0", "4097", "0x1002"]}), [4097, 4098])
        for report in ({}, None, {"err_len": 1, "err": []}, {"err_len": 1, "err": ["unknown"]},
                       {"err_len": 1, "err": [None]}, {"err_len": -1, "err": []}):
            with self.subTest(report=report), self.assertRaises(ValueError):
                active_robot_errors(report)

    def test_signal_cleanup_precedes_ros_shutdown(self):
        original_destroy = DualArmTeleopNode.destroy_node
        for signum in (signal.SIGINT, signal.SIGTERM):
            contexts, controllers = [], []
            previous = signal.getsignal(signum)
            def make_controller(_, side, ip, port):
                controller = FakeController(side)
                controllers.append(controller)
                return controller
            def destroy(node):
                contexts.append(node.context.ok())
                return original_destroy(node)
            def interrupt(*args, **kwargs):
                signal.raise_signal(signum)
            with patch.object(DualArmTeleopNode, "_make_controller", make_controller), \
                 patch.object(DualArmTeleopNode, "destroy_node", destroy), \
                 patch("dual_arm_teleop_node.rclpy.spin_once", side_effect=interrupt):
                arm_main(["--ros-args", "-p", "enabled_sides:=[left]", "-p", "telemetry_enabled:=true",
                          "-p", "telemetry_host_ip:=127.0.0.1"])
            self.assertEqual(contexts, [True])
            self.assertFalse(rclpy.ok())
            self.assertEqual(signal.getsignal(signum), previous)
            self.assertEqual(controllers[0].robot.settings[-1], controllers[0].robot.original)

    def test_single_arm_telemetry_skill_services_and_recording(self):
        rclpy.init(args=["--ros-args", "-p", "enabled_sides:=[left]", "-p", "telemetry_enabled:=true",
                         "-p", "telemetry_host_ip:=127.0.0.1"])
        executor = SingleThreadedExecutor()
        nodes = []
        with tempfile.TemporaryDirectory() as temp:
            try:
                with patch.object(DualArmTeleopNode, "_make_controller", lambda _, side, ip, port: FakeController(side)):
                    arm = DualArmTeleopNode()
                nodes.append(arm)
                self.assertEqual(set(arm.controllers), {"left"})
                arm.homes["left"].path = Path(temp)/"home.json"
                sdk = arm.controllers["left"].robot
                hand = HandSkillNode(SimpleNamespace(policy_module=ROOT/"dexscrew/xhand-deploy/nut_policy.py",
                    checkpoint=ROOT/"dexscrew/xhand-deploy/checkpoints/nutbolt_1019.pt",
                    release_file=Path(temp)/"release.json", serial_port="SIM", baud_rate=3000000,
                    hand_id=22, hardware=False, hz=20))
                nodes.append(hand)
                with patch("skill_teleop.collect.CameraProducer", FakeCamera):
                    collector = Collector(SimpleNamespace(side="left", data_dir=Path(temp)/"data",
                        camera_serial="SIM", max_age_ms=100, allow_simulated=True))
                nodes.append(collector)
                operator = rclpy.create_node("offline_test_operator")
                nodes.append(operator)
                for node in nodes: executor.add_node(node)
                send_arm_commands = False
                camera_seq, next_source = 0, 0

                def pump(seconds):
                    nonlocal camera_seq, next_source
                    until = time.monotonic()+seconds
                    while time.monotonic() < until:
                        now = time.monotonic()
                        if now >= next_source:
                            sdk.push()
                            if send_arm_commands:
                                msg = Pose(); msg.position.x=.2; msg.orientation.w=1.
                                arm.sessions["left"].handle_delta_pose(msg)
                            collector.receive(dict(stream="camera", seq=camera_seq, t_ns=time.monotonic_ns(),
                                valid=True, metadata={"serial":"SIM", "depth_scale":.001},
                                rgb=np.zeros((24,32,3),np.uint8), depth=np.ones((24,32),np.uint16)))
                            camera_seq += 1
                            next_source = now+.05
                        executor.spin_once(timeout_sec=.002)

                def service(name, service_type=Trigger):
                    client=operator.create_client(service_type,name)
                    self.assertTrue(client.wait_for_service(timeout_sec=2))
                    future=client.call_async(service_type.Request())
                    for _ in range(100):
                        pump(.01)
                        if future.done(): break
                    self.assertTrue(future.done())
                    response=future.result()
                    operator.destroy_client(client)
                    return response

                pump(.5)
                self.assertTrue(collector.latest["arm_state"]["valid"])
                self.assertEqual(arm.controllers["left"].commands, [])
                self.assertIn("arm_command", collector.readiness())
                self.assertFalse(service("/collection/start").success)
                self.assertFalse(service("/collection/success").success)
                self.assertFalse(service("/left/home").success)  # No taught home; no movement.
                arm.sessions["left"].start()
                send_arm_commands = True
                pump(.15)
                self.assertIsNone(collector.readiness())
                np.testing.assert_array_equal(hand.release, hand.policy.initial_hardware)
                self.assertTrue(service("/xhand/skill/release").success)
                np.testing.assert_array_equal(hand.transition[2], hand.policy.initial_hardware)
                pump(1.3)
                np.testing.assert_allclose(hand.target, hand.policy.initial_hardware)
                self.assertEqual(hand.mode, "HOLD")
                self.assertTrue(service("/collection/start").success)
                self.assertTrue(service("/xhand/skill/grasp").success)
                pump(1.3)
                self.assertTrue(hand.ready)
                self.assertTrue(service("/xhand/skill/start").success)
                pump(.5)
                self.assertTrue(service("/xhand/skill/stop").success)
                self.assertTrue(service("/collection/success").success)
                episode=next((Path(temp)/"data").iterdir())
                meta, streams=read_episode(episode)
                self.assertEqual(meta["status"],"success")
                self.assertTrue(all(r["valid"] for r in streams["arm_state"]))
                self.assertTrue(any(r["mode"]=="SCREW" for r in streams["hand_command"]))
                self.assertTrue(all(len(r["positions"])==6 for r in streams["arm_state"]))
                self.assertFalse(service("/left/capture_home").success)  # Teleop still active.
                send_arm_commands = False
                self.assertTrue(service("/left/stop_teleop", SetBool).success)
                self.assertTrue(service("/left/capture_home").success)
                sdk.joints = [0.] * 6
                self.assertTrue(service("/left/home").success)
                self.assertEqual(sdk.home_moves, [([10.] * 6, 10, 0, 0, 0)])
                self.assertFalse(service("/left/start_teleop", SetBool).success)
                count = len(arm.controllers["left"].commands)
                arm.sessions["left"].handle_delta_pose(Pose())
                self.assertEqual(len(arm.controllers["left"].commands), count)
                sdk.joints, sdk.trajectory_type = [10.] * 6, 0
                pump(.7)
                self.assertEqual(arm.homes["left"].phase, "ARRIVED")
                self.assertIsNone(arm.sessions["left"].motion_guard)
                sdk.errors = [0, 4097]
                pump(.15)
                self.assertFalse(collector.latest["arm_state"]["valid"])
                self.assertIn("4097", collector.latest["arm_state"]["error"])
                sdk.errors = [0]
                pump(.15)
                self.assertTrue(collector.latest["arm_state"]["valid"])
                # Reconnection invalidates telemetry, without silently carrying old frames.
                arm.controllers["left"].connection_generation += 1
                pump(.15)
                self.assertFalse(collector.latest["arm_state"]["valid"])
            finally:
                for node in reversed(nodes):
                    executor.remove_node(node)
                    node.destroy_node()
                executor.shutdown()
                rclpy.shutdown()
            self.assertEqual(sdk.settings[-1], sdk.original)


if __name__ == "__main__":
    unittest.main()
