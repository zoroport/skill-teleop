"""One D435 producer; SDK imported only when its thread is started."""
import threading
import time

import numpy as np


class CameraProducer:
    def __init__(self, callback, serial="", width=640, height=480, fps=30):
        self.callback = callback
        self.serial, self.width, self.height, self.fps = serial, width, height, fps
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)

    def _run(self):
        pipeline, seq = None, 0
        try:
            import pyrealsense2 as rs
            devices = list(rs.context().query_devices())
            serial = self.serial
            if not serial:
                if len(devices) != 1:
                    raise RuntimeError(f"Found {len(devices)} cameras; set camera_serial explicitly")
                serial = devices[0].get_info(rs.camera_info.serial_number)
            pipeline, config = rs.pipeline(), rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            profile = pipeline.start(config)
            alignment = rs.align(rs.stream.color)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            metadata = {"serial": serial, "mount": "fixed_external", "width": self.width,
                        "height": self.height, "fps": self.fps,
                        "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale(),
                        "intrinsics": {k: getattr(intr, k) for k in ("width", "height", "fx", "fy", "ppx", "ppy")},
                        "distortion_model": str(intr.model), "distortion_coeffs": list(intr.coeffs),
                        "depth_aligned_to": "color", "rgb_encoding": "rgb8"}
            # Warm up auto exposure before making frames available for recording.
            for _ in range(30):
                if self.stop_event.is_set():
                    return
                pipeline.wait_for_frames(1000)
            previous_numbers = None
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(1000)
                received = time.monotonic_ns()
                aligned = alignment.process(frames)
                color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                if not color or not depth:
                    raise RuntimeError("Missing color or depth frame")
                numbers = (color.get_frame_number(), depth.get_frame_number())
                deltas = (None if previous_numbers is None else
                          [current - previous for current, previous in zip(numbers, previous_numbers)])
                gap = previous_numbers is not None and any(
                    current != previous + 1 for current, previous in zip(numbers, previous_numbers))
                previous_numbers = numbers
                self.callback({"stream": "camera", "t_ns": received, "seq": seq,
                               "valid": not gap, "error": "Camera device frame gap" if gap else "",
                               "frame_gap": gap, "frame_deltas": deltas,
                               "metadata": metadata,
                               "device_timestamp_ms": color.get_timestamp(),
                               "depth_timestamp_ms": depth.get_timestamp(),
                               "timestamp_domain": str(color.get_frame_timestamp_domain()),
                               "depth_timestamp_domain": str(depth.get_frame_timestamp_domain()),
                               "frame_number": numbers[0], "depth_frame_number": numbers[1],
                               "rgb": np.asanyarray(color.get_data()).copy(),
                               "depth": np.asanyarray(depth.get_data()).copy()})
                seq += 1
        except Exception as exc:
            self.callback({"stream": "camera", "t_ns": time.monotonic_ns(), "seq": seq,
                           "valid": False, "error": str(exc)})
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
