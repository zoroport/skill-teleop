"""Bounded asynchronous episode writer. Incomplete episodes never look successful."""
import copy
import json
import os
import pickle
import queue
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from .schema import ACTION_SPEC, SCHEMA_VERSION, atomic_json, clock_identity


class EpisodeWriter:
    def __init__(self, root, metadata=None, queue_size=128, chunk_size=32):
        if queue_size < 1 or chunk_size < 1:
            raise ValueError("queue_size and chunk_size must be positive")
        self.path = Path(root) / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        (self.path / "chunks").mkdir(parents=True, exist_ok=False)
        self.metadata = {"schema_version": SCHEMA_VERSION, "action_spec": ACTION_SPEC,
                         "episode_id": self.path.name, "status": "recording",
                         "start_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns(),
                         "clock": clock_identity(), "config": metadata or {}}
        atomic_json(self.path / "metadata.json", self.metadata)
        self.queue = queue.Queue(maxsize=queue_size)
        self.chunk_size = chunk_size
        self.error = None
        self.closed = False
        self.counts = Counter()
        self.invalid = Counter()
        self.last_time = {}
        self.intervals = {}
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def submit(self, record):
        if self.closed or self.error:
            return False
        try:
            self.queue.put_nowait(copy.deepcopy(record))
            return True
        except queue.Full:
            self.error = "writer queue overflow; episode is incomplete"
            return False

    def _flush(self, chunk, index):
        dest = self.path / "chunks" / f"{index:06d}.pkl"
        temp = dest.with_suffix(".tmp")
        with temp.open("wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, dest)

    def _write(self):
        chunk, index = [], 0
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    if chunk:
                        self._flush(chunk, index)
                    return
                chunk.append(item)
                stream = item["stream"]
                self.counts[stream] += 1
                if not item.get("valid", False):
                    self.invalid[stream] += 1
                if stream in self.last_time:
                    self.intervals.setdefault(stream, []).append((item["t_ns"]-self.last_time[stream])/1e6)
                self.last_time[stream] = item["t_ns"]
                if len(chunk) >= self.chunk_size:
                    self._flush(chunk, index)
                    chunk, index = [], index + 1
        except Exception as exc:
            self.error = f"writer failed: {exc}"

    def finish(self, status="interrupted"):
        if self.closed:
            return self.path
        if status not in ("success", "failed", "interrupted"):
            raise ValueError(status)
        self.closed = True
        end = time.monotonic_ns()
        deadline = time.monotonic() + 10
        while self.thread.is_alive() and time.monotonic() < deadline:
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.thread.is_alive():
            self.error = "writer did not finish within 10 seconds"
        self.metadata.update(end_ns=end, status="incomplete" if self.error else status,
                             error=self.error, counts=dict(self.counts))
        atomic_json(self.path / "metadata.json", self.metadata)
        timing = {}
        for name, intervals in list(self.intervals.items()):
            dt = np.asarray(intervals)
            timing[name] = {"mean_ms": float(dt.mean()), "p95_ms": float(np.percentile(dt, 95)),
                            "max_ms": float(dt.max()), "nonpositive_intervals": int((dt<=0).sum())}
        atomic_json(self.path / "quality.json", {"status": self.metadata["status"],
                    "counts": dict(self.counts), "invalid": dict(self.invalid),
                    "intervals": timing, "error": self.error})
        return self.path


def read_episode(path, lazy_camera=False):
    """Read only locally collected/trusted pickle files."""
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported schema version")
    streams = {}
    for chunk in sorted((path / "chunks").glob("*.pkl")):
        with chunk.open("rb") as f:
            for offset, record in enumerate(pickle.load(f)):
                if lazy_camera and record["stream"] == "camera":
                    record.pop("rgb", None)
                    record.pop("depth", None)
                    record["_chunk"] = str(chunk)
                    record["_offset"] = offset
                streams.setdefault(record["stream"], []).append(record)
    for records in streams.values():
        records.sort(key=lambda r: (r["t_ns"], r.get("seq", 0)))
    return metadata, streams
