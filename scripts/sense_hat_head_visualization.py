#!/usr/bin/env python3
"""Sense HAT IMU 3-D head visualization

Displays a textured sphere (human head) in a VPython window whose orientation
and (slight) translation follow live IMU data from a Sense HAT on a Raspberry Pi.

Requirements (install with pip):
    vpython sense-hat numpy

Run:
    python3 sense_hat_head_visualization.py
Press Ctrl-C to quit.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from sense_hat import SenseHat  # type: ignore
except ImportError as exc:
    sys.stderr.write(
        "Error: Sense HAT library not found. Install with\n"
        "    sudo apt-get install sense-hat\n"
    )
    raise

try:
    from vpython import canvas, vector, rate, sphere, textures  # type: ignore
except ImportError as exc:
    sys.stderr.write(
        "Error: vpython not installed. Install with 'pip install vpython'\n"
    )
    raise

# --- Helpers -----------------------------------------------------------------
GRAVITY = 9.80665  # m/s^2
ACCEL_SCALE = 0.05  # Translation scaling factor for visual effect

def euler_to_vpython(roll: float, pitch: float, yaw: float) -> Tuple[vector, vector]:
    """Convert roll/pitch/yaw (deg) to VPython axis and up vectors.

    VPython's `sphere` orientation is controlled by its `axis` (direction of +z)
    and `up` (direction of +y). We compute these from the rotation matrix.
    """
    # Convert degrees to radians
    r, p, y = np.radians([roll, pitch, yaw])

    # Rotation matrices Z (yaw) * Y (pitch) * X (roll)
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)

    # Combined rotation matrix R = Rz * Ry * Rx
    R = np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])

    # In VPython default, +z is out of the screen. We map body frame forward (x)
    # to VPython's +z. Thus axis = R * (1,0,0), up = R * (0,1,0)
    axis = vector(*R[:, 0])
    up = vector(*R[:, 1])
    return axis, up


def main() -> None:
    sh = SenseHat()
    sh.set_imu_config(True, True, True)  # Enable all IMU sensors

    scene = canvas(title="Sense HAT IMU – 3D Head", width=800, height=600)
    scene.background = vector(0.1, 0.1, 0.1)
    scene.forward = vector(0, -0.2, -1)

    head = sphere(
        radius=0.5,
        texture=textures.earth,  # Placeholder texture; earth looks like a head-ish
        make_trail=False,
    )

    # Simple integration variables for translation demo
    velocity = np.zeros(3)
    position = np.zeros(3)
    last_time = time.time()

    while True:
        rate(60)  # Limit to 60 FPS
        now = time.time()
        dt = now - last_time
        last_time = now

        # Orientation (degrees)
        ori = sh.get_orientation()
        axis, up = euler_to_vpython(ori["roll"], ori["pitch"], ori["yaw"])
        head.axis = axis
        head.up = up

        # Acceleration (g). Convert to m/s^2 and remove gravity by subtracting
        # body-frame Z component transformed to world frame. For demo, we just
        # integrate raw accel for a fun effect.
        accel = sh.get_accelerometer_raw()
        a_world = np.array([accel["x"], accel["y"], accel["z"]]) * GRAVITY
        velocity += a_world * dt
        position += velocity * dt

        # Clamp position to avoid drifting away
        position *= 0.99  # damping
        head.pos = vector(*(position * ACCEL_SCALE))

        # Show brief text overlay
        scene.caption = (
            f"{datetime.utcnow().isoformat()}\n"
            f"Roll={ori['roll']:.1f}° Pitch={ori['pitch']:.1f}° Yaw={ori['yaw']:.1f}°\n"
            f"Accel x={accel['x']:+.2f} y={accel['y']:+.2f} z={accel['z']:+.2f} g"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
