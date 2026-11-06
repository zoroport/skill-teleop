import copy
import json
from pathlib import Path
import pickle
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from skill_teleop.convert import align_episode, convert
from skill_teleop.schema import ACTION_SPEC, action_to_pose, pose_to_action, atomic_json
from skill_teleop.storage import EpisodeWriter, read_episode
from skill_teleop.hand_device import HandDevice, HandSDKError
from skill_teleop.camera import CameraProducer
from types import SimpleNamespace


def fixture(dof=6, frames=40, tactile=True):
    streams = {name: [] for name in ("arm_state", "arm_command", "hand_state", "hand_command", "camera")}
    for i in range(frames):
        t = 1_000_000_000 + i * 50_000_000
        pose = [.3, .1, .4, 0, 0, 0, 1]
        common = dict(t_ns=t, seq=i, valid=True, session=1, frame_id="work", error="")
        streams["arm_state"].append(dict(common, positions=[.1] * dof, velocities=[0.] * dof,
            pose=pose, joint_names=[f"j{k}" for k in range(dof)],
            metadata={"work_frame": {"name": "base"}, "tool_frame": {"name": "tool"}}))
        # Arm remains stationary while the hand actively screws.
        streams["arm_command"].append(dict(common, t_ns=t+10_000_000, target=pose, kind="command"))
        streams["hand_state"].append(dict(common, positions=[i*.01]*12, joint_names=[f"h{k}" for k in range(12)],
            tactile_valid=tactile, tactile=[float(i)]*30 if tactile else [], tactile_shape=[5,2,3] if tactile else []))
        streams["hand_command"].append(dict(common, t_ns=t+20_000_000, positions=[(i+1)*.01]*12, mode="SCREW"))
        streams["camera"].append(dict(common, metadata={"serial": "fake", "depth_scale": .001},
            rgb=np.full((24,32,3), i, dtype=np.uint8), depth=np.full((24,32), 1000, dtype=np.uint16)))
    meta = {"schema_version": 1, "episode_id": "fake", "start_ns": 1_050_000_000,
            "end_ns": 1_000_000_000 + frames*50_000_000, "status": "success", "action_spec": ACTION_SPEC}
    return meta, streams


def write_fixture(path, dof=6, frames=40):
    meta, streams = fixture(dof, frames)
    meta["episode_id"] = path.name
    (path / "chunks").mkdir(parents=True)
    atomic_json(path / "metadata.json", meta)
    with (path / "chunks/000000.pkl").open("wb") as f:
        pickle.dump([dict(r, stream=k) for k, rows in streams.items() for r in rows], f)


class CollectionTests(unittest.TestCase):
    def test_camera_device_gap_is_not_a_valid_continuous_stream(self):
        class Frame:
            def __init__(self, number, depth=False):
                self.number, self.depth = number, depth
            def get_frame_number(self): return self.number
            def get_timestamp(self): return self.number * 1000 / 30
            def get_frame_timestamp_domain(self): return "hardware_clock"
            def get_data(self):
                return np.zeros((4, 6) if self.depth else (4, 6, 3), dtype=np.uint16 if self.depth else np.uint8)
        frames = iter([None]*30 + [SimpleNamespace(get_color_frame=lambda n=n: Frame(n),
                     get_depth_frame=lambda n=n: Frame(n, True)) for n in (100, 102, 103)])
        intr = SimpleNamespace(width=6, height=4, fx=2., fy=2., ppx=3., ppy=2., model="none", coeffs=[0.]*5)
        profile = SimpleNamespace(
            get_stream=lambda _: SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intr)),
            get_device=lambda: SimpleNamespace(first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: .001)))
        pipeline = SimpleNamespace(start=lambda _: profile, wait_for_frames=lambda _: next(frames), stop=lambda: None)
        config = SimpleNamespace(enable_device=lambda _: None, enable_stream=lambda *args: None)
        sdk = SimpleNamespace(context=lambda: SimpleNamespace(query_devices=lambda: []),
            pipeline=lambda: pipeline, config=lambda: config, align=lambda _: SimpleNamespace(process=lambda x: x),
            stream=SimpleNamespace(color=0, depth=1), format=SimpleNamespace(rgb8=0, z16=1))
        records = []
        def receive(record):
            records.append(record)
            if len(records) == 3: camera.stop_event.set()
        camera = CameraProducer(receive, serial="fake")
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            camera._run()
        self.assertEqual([r["valid"] for r in records], [True, False, True])
        self.assertEqual(records[1]["depth_frame_number"], 102)
        self.assertEqual(records[0]["depth"].dtype, np.uint16)

    def test_hand_sdk_failure_keeps_return_code(self):
        with self.assertRaises(HandSDKError) as raised:
            HandDevice.check(SimpleNamespace(error_code=17, error_message="timeout"), "send_command")
        self.assertEqual(raised.exception.return_code, 17)
        HandDevice.check(SimpleNamespace(error_code=0), "send_command")

    def test_finish_timeout_covers_full_writer_queue(self):
        gate, started = threading.Event(), threading.Event()
        original = EpisodeWriter._flush
        def stuck(writer, chunk, index):
            started.set()
            gate.wait(3)
            original(writer, chunk, index)
        with tempfile.TemporaryDirectory() as root, patch.object(EpisodeWriter, "_flush", stuck):
            w = EpisodeWriter(root, queue_size=1, chunk_size=1)
            try:
                w.submit(dict(stream="event", t_ns=0))
                self.assertTrue(started.wait(1))
                w.submit(dict(stream="event", t_ns=1))
                with patch("skill_teleop.storage.time.monotonic", side_effect=[0., 11., 11.]):
                    w.finish("success")
                self.assertEqual(w.metadata["status"], "incomplete")
                self.assertIn("10 seconds", w.error)
            finally:
                gate.set()
                w.queue.put(None, timeout=3)
                w.thread.join(timeout=3)

    def test_rotation_roundtrip_and_degenerate(self):
        for r in Rotation.random(25, random_state=0):
            pose = np.r_[[.3, -.1, .7], r.as_quat()]
            recovered, hand = action_to_pose(pose_to_action(pose, np.arange(12)))
            np.testing.assert_allclose(Rotation.from_quat(recovered[3:]).as_matrix(), r.as_matrix(), atol=1e-6)
            np.testing.assert_allclose(hand, np.arange(12))
        with self.assertRaises(ValueError):
            action_to_pose(np.zeros(21))

    def test_causal_observation_and_next_action_static_arm(self):
        for dof in (6, 7):
            meta, streams = fixture(dof)
            segments, quality = align_episode(meta, streams, tactile=True)
            self.assertEqual(len(segments), 1)
            self.assertEqual(len(segments[0]), 39)
            row = segments[0][0]
            self.assertEqual(row["joint_positions"].size, dof)
            self.assertEqual(row["base_rgb"].shape, (1,240,320,3))
            self.assertEqual(row["base_depth"].dtype, np.uint16)
            self.assertEqual(row["action"].shape, (21,))
            self.assertAlmostEqual(row["xhand_pos"][0], .01)
            self.assertAlmostEqual(row["action"][-1], .02)
            self.assertLessEqual(row["source_times_ns"]["camera"], row["timestamp_ns"])
            self.assertGreater(row["source_times_ns"]["hand_command"], row["timestamp_ns"])

    def test_failure_and_packet_gap_split_sequences(self):
        meta, streams = fixture()
        streams["arm_command"][15]["valid"] = False
        del streams["camera"][25]
        segments, quality = align_episode(meta, streams)
        self.assertGreaterEqual(len(segments), 3)
        self.assertTrue(any("invalid" in k for k in quality))
        self.assertIn("sequence_or_session_gap", quality)

    def test_tactile_and_stale_or_session_rejection(self):
        meta, streams = fixture(tactile=False)
        self.assertTrue(align_episode(meta, streams)[0])
        self.assertFalse(align_episode(meta, streams, tactile=True)[0])
        streams["camera"] = streams["camera"][:2]
        segments, quality = align_episode(meta, streams)
        self.assertIn("stale_observation", quality)
        streams["hand_command"] = [dict(r, session=2) for r in streams["hand_command"]]
        self.assertFalse(align_episode(meta, streams)[0])

    def test_latest_failed_command_never_becomes_a_hold(self):
        meta, streams = fixture()
        streams["arm_command"] = streams["arm_command"][:10]
        streams["arm_command"][-1]["valid"] = False
        segments, _ = align_episode(meta, streams)
        self.assertLess(max(r["timestamp_ns"] for s in segments for r in s), streams["arm_command"][-1]["t_ns"])

    def test_writer_success_interruption_and_disk_error(self):
        with tempfile.TemporaryDirectory() as root:
            w = EpisodeWriter(root, chunk_size=2)
            for i in range(5):
                self.assertTrue(w.submit(dict(stream="event", t_ns=i, seq=i, valid=True)))
            path = w.finish("success")
            meta, streams = read_episode(path)
            self.assertEqual(meta["status"], "success")
            self.assertEqual(len(streams["event"]), 5)
            w = EpisodeWriter(root)
            self.assertEqual(json.loads((w.finish() / "metadata.json").read_text())["status"], "interrupted")
            with patch.object(EpisodeWriter, "_flush", side_effect=OSError("disk full")):
                w = EpisodeWriter(root, chunk_size=1)
                w.submit(dict(stream="event", t_ns=0))
                w.finish("success")
                self.assertEqual(w.metadata["status"], "incomplete")

    def test_overflow_is_visible_and_cannot_succeed(self):
        gate = threading.Event()
        started = threading.Event()
        original = EpisodeWriter._flush
        def slow(writer, chunk, index):
            started.set()
            gate.wait(2)
            original(writer, chunk, index)
        with tempfile.TemporaryDirectory() as root, patch.object(EpisodeWriter, "_flush", slow):
            w = EpisodeWriter(root, queue_size=1, chunk_size=1)
            w.submit(dict(stream="event", t_ns=0))
            self.assertTrue(started.wait(1))
            w.submit(dict(stream="event", t_ns=1))
            self.assertFalse(w.submit(dict(stream="event", t_ns=2)))
            gate.set()
            w.finish("success")
            self.assertEqual(w.metadata["status"], "incomplete")

    def test_conversion_split_manifest_and_failure_exclusion(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for i in range(3):
                write_fixture(root / "raw" / f"episode_{i}")
            bad = root / "raw/episode_2/metadata.json"
            meta = json.loads(bad.read_text()); meta["status"] = "failed"; atomic_json(bad, meta)
            result = convert(root / "raw", root / "dataset", tactile=True)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["tactile_dim"], 30)
            self.assertEqual(len(result["episodes"]), 2)
            self.assertEqual({e["split"] for e in result["episodes"]}, {"train", "test"})
            self.assertEqual(len({e["id"] for e in result["episodes"]}), 2)


if __name__ == "__main__":
    unittest.main()
