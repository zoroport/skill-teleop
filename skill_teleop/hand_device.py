"""One-read-per-cycle RS485 device adapter with an explicit simulation backend."""
import fcntl
import hashlib
import os
from pathlib import Path

import numpy as np


class HandSDKError(RuntimeError):
    def __init__(self, operation, result):
        self.return_code = int(getattr(result, "error_code", -1))
        super().__init__(f"{operation}: code={self.return_code}, {getattr(result, 'error_message', result)}")


class HandDevice:
    def __init__(self, serial_port, baud_rate, hand_id, hardware=False):
        self.serial_port, self.baud_rate, self.hand_id = serial_port, baud_rate, hand_id
        self.hardware = hardware
        self.device = self.lock = None
        self.measured = np.zeros(12)

    @staticmethod
    def check(result, operation):
        if result is None or int(getattr(result, "error_code", -1)) != 0:
            raise HandSDKError(operation, result)

    def open(self):
        if not self.hardware:
            return
        from xhand_controller import xhand_control as sdk
        key = hashlib.sha256(os.path.realpath(self.serial_port).encode()).hexdigest()[:16]
        self.lock = open(Path("/tmp") / f"skill_teleop_hand_{key}.lock", "a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.device = sdk.XHandControl()
            self.check(self.device.open_serial(self.serial_port, self.baud_rate), "open_serial")
            self.command = sdk.HandCommand_t()
            for i in range(12):
                command = self.command.finger_command[i]
                command.id, command.mode = i, 3
                command.kp, command.ki, command.kd, command.tor_max = 100, 0, 1, 100
        except BaseException:
            self.close()
            raise

    def read(self):
        if not self.hardware:
            return self.measured.copy(), None
        response, state = self.device.read_state(self.hand_id, True)
        self.check(response, "read_state")
        positions = np.array([state.finger_state[i].position for i in range(12)], dtype=float)
        if not np.isfinite(positions).all():
            raise ValueError("Nonfinite joint feedback")
        tactile = None
        try:
            sensors = state.sensor_data
            force = np.asarray([[[f.fx, f.fy, f.fz] for f in s.raw_force] for s in sensors], dtype=float)
            if force.ndim == 3 and force.shape[0] == 5 and force.shape[2] == 3 and force.size and np.isfinite(force).all():
                tactile = force
        except (AttributeError, TypeError, ValueError):
            pass  # Missing tactile is explicitly represented by None, not zeros.
        return positions, tactile

    def send(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (12,) or not np.isfinite(target).all():
            raise ValueError("Invalid hand target")
        if not self.hardware:
            self.measured = target.copy()
            return
        for i, value in enumerate(target):
            self.command.finger_command[i].position = float(value)
        self.check(self.device.send_command(self.hand_id, self.command), "send_command")

    def close(self):
        try:
            if self.device is not None:
                self.device.close_device()
        finally:
            self.device = None
            if self.lock is not None:
                self.lock.close()
                self.lock = None
