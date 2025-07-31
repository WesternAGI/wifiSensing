#!/usr/bin/env python3
"""
High-Rate Active STA for WiFi CSI Logging with Live Monitoring
"""

import os
import time
import argparse
import subprocess
import logging
import serial
import socket
import threading
from datetime import datetime

# -------------------- CONFIG --------------------
WIFI_INTERFACE = "wlan0"
WIFI_CONNECT_TIMEOUT = 10
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
MAX_PACKET_RATE = 1000  # packets/sec
MONITOR_INTERVAL = 1.0  # seconds

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("active_sta.log", mode="a")
    ]
)
logger = logging.getLogger("active_sta")

# Global counters for live monitoring
packet_counter = 0
csi_counter = 0
stop_event = threading.Event()

# -------------------- UTILITIES --------------------
def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        return ""

def optimize_wifi(interface):
    """Disable power save and increase performance for max CSI."""
    logger.info("Optimizing WiFi interface for max throughput...")
    os.system(f"sudo iw dev {interface} set power_save off")
    os.system(f"sudo iwconfig {interface} txpower auto")
    os.system(f"sudo ifconfig {interface} txqueuelen 2000")  # increase tx queue
    os.system(f"sudo iw {interface} set bitrates legacy-2.4 54")  # force max rate

def parse_status(status_str):
    status = {}
    for line in status_str.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            status[k.strip()] = v.strip()
    return status

# -------------------- PACKET TRANSMISSION --------------------
def send_packets_continuously(target_ip, packet_rate, stop_event):
    global packet_counter
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # increase buffer
    payload = b"CSI_TRIGGER_PACKET_" + b"A" * 100

    logger.info(f"Sending packets to {target_ip} at {packet_rate} pkt/s")
    interval = 1.0 / packet_rate
    next_time = time.time()

    while not stop_event.is_set():
        try:
            sock.sendto(payload, (target_ip, 80))
            packet_counter += 1
            next_time += interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.time()
        except Exception as e:
            logger.error(f"Packet send error: {e}")
            time.sleep(0.001)

    sock.close()
    logger.info("Packet transmission stopped.")

# -------------------- SERIAL LOGGING --------------------
def log_serial_data(output_file, duration, skip_serial=False):
    global csi_counter
    start_time = time.time()
    if skip_serial:
        logger.info("Skipping serial logging (mock mode).")
        time.sleep(duration)
        return

    try:
        with serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01) as ser, open(output_file, "w") as f:
            f.write("timestamp,csi_data\n")
            while time.time() - start_time < duration and not stop_event.is_set():
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if "CSI_DATA" in line:
                        csi_counter += 1
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        f.write(f"{timestamp},{line}\n")
                        f.flush()
                else:
                    time.sleep(0.001)
    except Exception as e:
        logger.error(f"Serial logging error: {e}")

# -------------------- LIVE MONITOR --------------------
def live_monitor():
    global packet_counter, csi_counter
    last_packets = 0
    last_csi = 0
    while not stop_event.is_set():
        time.sleep(MONITOR_INTERVAL)
        pps = packet_counter - last_packets
        cps = csi_counter - last_csi
        logger.info(f"Live: {pps} pkt/s | {cps} CSI/s | Total packets={packet_counter}, CSI={csi_counter}")
        last_packets, last_csi = packet_counter, csi_counter

# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssid", type=str, default="mywifi_ssid")
    parser.add_argument("--password", type=str, default="mywifi_pass")
    parser.add_argument("--output", type=str, default="output.csv")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--rate", type=int, default=MAX_PACKET_RATE)
    parser.add_argument("--interface", type=str, default=WIFI_INTERFACE)
    parser.add_argument("--no-serial", action="store_true")
    args = parser.parse_args()

    optimize_wifi(args.interface)
    gateway_ip = "192.168.4.1"  # or auto-detect gateway

    # Start threads
    threads = [
        threading.Thread(target=send_packets_continuously, args=(gateway_ip, args.rate, stop_event)),
        threading.Thread(target=log_serial_data, args=(args.output, args.duration, args.no_serial)),
        threading.Thread(target=live_monitor)
    ]

    for t in threads: t.start()
    time.sleep(args.duration)
    stop_event.set()
    for t in threads: t.join()

    logger.info("Capture complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        logger.info("Interrupted by user.")
