"""Inspect a raw episode, including interrupted recordings, without changing it."""
import argparse
from collections import Counter
import json

import numpy as np

from .storage import read_episode


def inspect_episode(path):
    metadata, streams = read_episode(path, lazy_camera=True)
    report = {"episode": metadata["episode_id"], "status": metadata["status"], "streams": {}}
    for name, records in streams.items():
        times = np.asarray([r["t_ns"] for r in records], dtype=np.int64)
        dt = np.diff(times) / 1e6
        gaps = sum(b.get("seq", 0) != a.get("seq", -1) + 1 for a, b in zip(records, records[1:]))
        info = {"records": len(records), "invalid": sum(not r.get("valid", False) for r in records),
                "sequence_gaps": gaps,
                "errors": dict(Counter(r.get("error", "") for r in records if not r.get("valid", False)))}
        if len(dt):
            info["interval_ms"] = {"mean": float(dt.mean()), "p95": float(np.percentile(dt, 95)),
                                   "max": float(dt.max())}
        report["streams"][name] = info
    report["stages"] = [{"t_ns": r["t_ns"], "stage": r["stage"]} for r in streams.get("event", [])]
    report["note"] = "Host timestamps and SDK success do not prove physical actuator arrival."
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    args = parser.parse_args()
    print(json.dumps(inspect_episode(args.episode), indent=2))
