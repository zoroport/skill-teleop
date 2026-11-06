"""Check collection imports, model/release paths and ROS interfaces without opening devices."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("arm", "hand", "collector", "all"), default="all")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parents[2]/"dexscrew/xhand-deploy/checkpoints/nutbolt_1019.pt")
    parser.add_argument("--release-file", type=Path, default=Path("calibration/xhand22_release.json"))
    args = parser.parse_args()
    modules = ["numpy", "scipy", "rclpy", "skill_teleop_interfaces.msg"]
    if args.component in ("arm", "all"):
        modules += ["Robotic_Arm.rm_robot_interface"]
    if args.component in ("hand", "all"):
        modules += ["torch", "xhand_controller.xhand_control"]
    if args.component in ("collector", "all"):
        modules += ["cv2", "pyrealsense2"]
    errors = []
    print(f"Python: {sys.executable}\nNo devices will be opened.")
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"OK {name}")
        except Exception as exc:
            errors.append(name)
            print(f"FAIL {name}: {exc}")
    if args.component in ("hand", "all"):
        if not args.checkpoint.is_file():
            errors.append("checkpoint")
            print(f"FAIL checkpoint: {args.checkpoint}")
        else:
            print(f"OK checkpoint exists: {args.checkpoint}")
        if args.release_file.is_file():
            try:
                data = json.loads(args.release_file.read_text())
                from .schema import vector
                vector(data["positions"], 12)
                if data["order"] != "hardware":
                    raise ValueError("order must be hardware")
                print(f"OK release calibration for hand {data['hand_id']}")
            except Exception as exc:
                errors.append("release calibration")
                print(f"FAIL release calibration: {exc}")
        else:
            print("INFO no release override; release uses the nut policy initial pose")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
