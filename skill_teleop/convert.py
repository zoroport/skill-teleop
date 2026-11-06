"""Offline conversion: python -m skill_teleop.convert RAW_ROOT DATASET_ROOT."""
import argparse
import bisect
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .schema import ACTION_SPEC, atomic_json, pose_to_action, vector
from .storage import read_episode


class Timeline:
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda x: x["t_ns"])
        self.times = [x["t_ns"] for x in self.rows]
        self.breaks = []
        for a, b in zip(self.rows, self.rows[1:]):
            if (a.get("session") != b.get("session") or
                    ("seq" in a and "seq" in b and b["seq"] != a["seq"] + 1)):
                self.breaks.append(b["t_ns"])

    def latest(self, t, inclusive=True):
        index = (bisect.bisect_right if inclusive else bisect.bisect_left)(self.times, t) - 1
        return self.rows[index] if index >= 0 else None

    def interval(self, start, end):
        return self.rows[bisect.bisect_left(self.times, start):bisect.bisect_left(self.times, end)]

    def broken(self, start, end):
        return bisect.bisect_left(self.breaks, end) > bisect.bisect_left(self.breaks, start)


def align_episode(metadata, streams, hz=20, max_age_ms=100, tactile=False):
    """Return contiguous segments. Observations are causal, actions use [t,t+dt)."""
    if not np.isfinite(hz) or hz <= 0 or max_age_ms <= 0:
        raise ValueError("Invalid resampling frequency/age")
    period = round(1e9 / hz)
    age = round(max_age_ms * 1e6)
    names = ("arm_state", "hand_state", "camera", "arm_command", "hand_command")
    lines = {name: Timeline(streams.get(name, [])) for name in names}
    events = Timeline(streams.get("event", []))
    report = Counter()
    segments, segment = [], []
    last_quat = None
    cached_chunk, camera_rows = None, None
    for t in range(metadata["start_ns"], metadata["end_ns"] - period + 1, period):
        try:
            obs = {name: lines[name].latest(t) for name in names[:3]}
            cmd = {name: lines[name].latest(t + period, inclusive=False) for name in names[3:]}
            if any(x is None for x in list(obs.values()) + list(cmd.values())):
                raise ValueError("missing_stream")
            if any(not x["valid"] for x in list(obs.values()) + list(cmd.values())):
                raise ValueError("invalid_stream")
            if any(t - x["t_ns"] > age for x in obs.values()):
                raise ValueError("stale_observation")
            # A short failure followed by recovery in the same grid cell still breaks it.
            for name, line in lines.items():
                if line.broken(t, t + period):
                    raise ValueError("sequence_or_session_gap")
                if any(not x["valid"] for x in line.interval(t, t + period)):
                    raise ValueError("invalid_interval")
            arm, hand, camera = obs["arm_state"], obs["hand_state"], obs["camera"]
            ac, hc = cmd["arm_command"], cmd["hand_command"]
            if arm["session"] != ac["session"] or hand["session"] != hc["session"]:
                raise ValueError("session_mismatch")
            if arm["frame_id"] != ac["frame_id"]:
                raise ValueError("frame_mismatch")
            q = vector(arm["positions"])
            if len(q) not in (6, 7):
                raise ValueError("arm_dof")
            pose = vector(arm["pose"], 7).copy()
            norm = np.linalg.norm(pose[3:])
            if norm < 1e-8:
                raise ValueError("invalid_quaternion")
            pose[3:] /= norm
            if last_quat is not None and np.dot(last_quat, pose[3:]) < 0:
                pose[3:] *= -1
            last_quat = pose[3:].copy()
            frame = {"timestamp_ns": t, "joint_positions": q.astype(np.float32),
                     "ee_pos_quat": pose.astype(np.float32),
                     "xhand_pos": vector(hand["positions"], 12).astype(np.float32),
                     "action": pose_to_action(ac["target"], hc["positions"]),
                     "xhand_rl_flag": hc["mode"] == "SCREW",
                     "arm_active": arm.get("active", False),
                     "source_times_ns": {k: v["t_ns"] for k, v in {**obs, **cmd}.items()},
                     "source_sequences": {k: v["seq"] for k, v in {**obs, **cmd}.items()},
                     "stage": (events.latest(t) or {}).get("stage", hc["mode"])}
            if len(arm.get("velocities", [])) == len(q):
                frame["joint_velocities"] = vector(arm["velocities"], len(q)).astype(np.float32)
            if hand.get("tactile_valid", False):
                values = vector(hand["tactile"])
                if values.size == 0 or values.size != np.prod(hand["tactile_shape"]):
                    raise ValueError("tactile_shape")
                frame["xhand_tactile"] = values.astype(np.float32)
            elif tactile:
                raise ValueError("tactile_missing")
            frame["tactile_valid"] = hand.get("tactile_valid", False)
            if "_chunk" in camera:
                if cached_chunk != camera["_chunk"]:
                    with open(camera["_chunk"], "rb") as f:
                        camera_rows = pickle.load(f)
                    cached_chunk = camera["_chunk"]
                camera = camera_rows[camera["_offset"]]
            rgb, depth = camera["rgb"], camera["depth"]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("rgb_shape_or_dtype")
            if depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
                raise ValueError("depth_shape_or_dtype")
            frame["base_rgb"] = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)[None]
            frame["base_depth"] = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_NEAREST)[None]
            segment.append(frame)
            report["kept_frames"] += 1
        except (ValueError, KeyError, TypeError) as exc:
            report[str(exc)] += 1
            if segment:
                segments.append(segment)
                segment = []
            last_quat = None
    if segment:
        segments.append(segment)
    return segments, dict(report)


def layout_from_streams(streams):
    arm = next(x for x in streams["arm_state"] if x["valid"])
    hand = next(x for x in streams["hand_state"] if x["valid"])
    camera = next(x for x in streams["camera"] if x["valid"])
    return {"arm_dof": len(arm["positions"]), "arm_joint_names": arm["joint_names"],
            "hand_joint_names": hand["joint_names"], "frame_id": arm["frame_id"],
            "work_frame": arm["metadata"]["work_frame"],
            "tool_frame": arm["metadata"]["tool_frame"],
            "camera": camera["metadata"], "action_spec": ACTION_SPEC}


def convert(root, destination, hz=20, max_age_ms=100, tactile=False, seed=0):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError("Choose a new output directory; datasets are never merged implicitly")
    episodes = []
    for path in sorted(root.glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        if meta.get("status") == "success":
            episodes.append(path.parent)
    if len(episodes) < 2:
        raise ValueError("Need at least two successful episodes for disjoint train/test splits")
    shuffled = episodes.copy()
    random.Random(seed).shuffle(shuffled)
    test_ids = {p.name for p in shuffled[:max(1, round(len(shuffled) * .1))]}
    destination.mkdir(parents=True)
    manifest = {"status": "converting", "hz": hz, "max_age_ms": max_age_ms, "seed": seed,
                "tactile_required": tactile, "action_spec": ACTION_SPEC, "episodes": []}
    atomic_json(destination / "dataset.json", manifest)
    layout, tactile_dim = None, None
    try:
        for episode in episodes:
            meta, streams = read_episode(episode, lazy_camera=True)
            current_layout = layout_from_streams(streams)
            if layout is not None and current_layout != layout:
                raise ValueError(f"Incompatible hardware/coordinate/camera layout: {episode}")
            layout = current_layout
            segments, report = align_episode(meta, streams, hz, max_age_ms, tactile)
            split = "test" if episode.name in test_ids else "train"
            written = []
            for index, frames in enumerate(segments):
                # With obs=2, pred=16, act=8 this is the shortest usable sequence.
                if len(frames) < 8:
                    report["short_segment_frames"] = report.get("short_segment_frames", 0) + len(frames)
                    continue
                path = destination / split / f"{episode.name}_segment_{index:03d}"
                path.mkdir(parents=True)
                for i, frame in enumerate(frames):
                    if tactile:
                        dim = frame["xhand_tactile"].size
                        if tactile_dim is not None and dim != tactile_dim:
                            raise ValueError("Tactile dimensions changed across frames")
                        tactile_dim = dim
                    with (path / f"step_{i:06d}.pkl").open("wb") as f:
                        pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
                atomic_json(path / "episode.json", {"source_episode": episode.name, "frames": len(frames)})
                written.append(str(path.relative_to(destination)))
            manifest["episodes"].append({"id": episode.name, "split": split,
                                          "segments": written, "quality": report})
        for split in ("train", "test"):
            if not any(e["split"] == split and e["segments"] for e in manifest["episodes"]):
                raise ValueError(f"No usable {split} sequences; collect more complete episodes")
        manifest.update(status="complete", layout=layout, tactile_dim=tactile_dim)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        atomic_json(destination / "dataset.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--hz", type=float, default=20)
    parser.add_argument("--max-age-ms", type=float, default=100)
    parser.add_argument("--tactile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = convert(args.source, args.destination, args.hz, args.max_age_ms, args.tactile, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
