#!/usr/bin/env python3
"""
Real-time Sense HAT sensor monitor for Raspberry Pi.
Prints all available sensor measurements (temperature, humidity, pressure,
orientation, acceleration, gyroscope, magnetometer) in real time.

Run:
    python3 sense_hat_realtime.py
Press Ctrl+C to stop.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from sense_hat import SenseHat  # type: ignore
except ImportError as exc:
    sys.stderr.write(
        "Error: The Sense HAT library is not installed. "
        "Install it with\n    sudo apt-get install sense-hat\n"
    )
    raise


def format_dict(d: dict[str, float]) -> str:
    """Format a dictionary of float readings into a compact string."""
    return " ".join(f"{k}={v:6.2f}" for k, v in d.items())


def read_sensors(sh: SenseHat) -> dict[str, float]:
    """Read all relevant Sense HAT sensors and return as a flat dict."""
    data: dict[str, float] = {}

    # Environmental sensors
    data["temp_c"] = sh.get_temperature()
    data["humidity_%"] = sh.get_humidity()
    data["pressure_hPa"] = sh.get_pressure()

    # Orientation (Euler angles)
    orientation = sh.get_orientation()
    data["roll_deg"] = orientation["roll"]
    data["pitch_deg"] = orientation["pitch"]
    data["yaw_deg"] = orientation["yaw"]

    # Raw accelerometer
    accel = sh.get_accelerometer_raw()
    data["accel_x_g"] = accel["x"]
    data["accel_y_g"] = accel["y"]
    data["accel_z_g"] = accel["z"]

    # Gyroscope
    gyro = sh.get_gyroscope_raw()
    data["gyro_x_dps"] = gyro["x"]
    data["gyro_y_dps"] = gyro["y"]
    data["gyro_z_dps"] = gyro["z"]

    # Magnetometer
    mag = sh.get_compass_raw()
    data["mag_x_uT"] = mag["x"]
    data["mag_y_uT"] = mag["y"]
    data["mag_z_uT"] = mag["z"]

    return data


def main() -> None:
    sh = SenseHat()
    sh.clear()  # Ensure LED matrix is off

    interval = 1.0  # seconds between reads
    try:
        while True:
            data = read_sensors(sh)
            timestamp = datetime.utcnow().isoformat()
            print(f"{timestamp} | {format_dict(data)}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping real-time Sense HAT monitor. Clearing LED matrix…")
        sh.clear()
        sys.exit(0)


if __name__ == "__main__":
    main()
