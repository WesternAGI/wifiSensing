#!/usr/bin/env python3
"""
Cross-Platform Active Station **with Windows support** for WiFi CSI Collection
==============================================================================
This script is a **stand-alone** version of `active_sta.py` that now supports
Windows hosts in addition to Linux, Raspberry Pi and macOS.  It does *not*
import the original module; all functionality (network management, packet
transmission, serial capture, live monitor, CLI) is fully self-contained.

Added Windows features
----------------------
1.  Operating-system detection recognises Windows and returns the identifier
    ``"windows"``.
2.  A native Windows connection path leverages ``netsh wlan`` to create a
    temporary XML profile and connect to the target SSID.
3.  Restoration logic reconnects to the original AP using ``netsh wlan``.
4.  Default Wi-Fi interface on Windows is set to **"Wi-Fi"**.

Aside from these additions, the behaviour, CLI arguments and logging are
identical to the original cross-platform implementation.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional

# ------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover – allow running without python-dotenv
    def load_dotenv():  # type: ignore
        pass

try:
    import serial  # type: ignore
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ------------------------------------------------------------
# Constants & globals
# ------------------------------------------------------------
WIFI_INTERFACE = "wlan0"
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
MONITOR_INTERVAL = 1.0  # seconds

packet_counter = 0
csi_counter = 0
stop_event = threading.Event()

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("active_sta.log", mode="a")],
)
logger = logging.getLogger("active_sta_win")

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def optimize_wifi(interface: str) -> None:
    """Best-effort Wi-Fi NIC optimisation for maximum CSI generation (Linux/macOS)."""
    logger.info("Optimising WiFi interface for maximum throughput …")
    try:
        os.system(f"sudo iw dev {interface} set power_save off 2>/dev/null")
        os.system(f"sudo iwconfig {interface} txpower auto 2>/dev/null")
        os.system(f"sudo ifconfig {interface} txqueuelen 2000 2>/dev/null")
        os.system(f"sudo iw {interface} set bitrates legacy-2.4 54 2>/dev/null")
    except Exception as exc:  # pragma: no cover
        logger.debug(f"WiFi optimisation failed: {exc}")

# ------------------------------------------------------------
# Network manager
# ------------------------------------------------------------
class NetworkManager:
    """Cross-platform WiFi connection manager supporting Linux, macOS, RPi & Windows."""

    def __init__(self) -> None:
        self.os_type = self._detect_os()
        self.original_network: Optional[str] = None
        logger.info(f"Detected OS: {self.os_type}")

    # --------------------------------------------------
    # OS detection
    # --------------------------------------------------
    def _detect_os(self) -> str:
        system = platform.system().lower()
        if system.startswith("win"):
            return "windows"
        if system == "darwin":
            return "macos"
        if system == "linux":
            # Raspberry Pi check
            try:
                with open("/proc/cpuinfo", "r") as fp:
                    cpuinfo = fp.read().lower()
                    if "raspberry pi" in cpuinfo or "bcm" in cpuinfo:
                        return "raspberry_pi"
            except FileNotFoundError:
                pass
            return "linux"
        raise OSError(f"Unsupported operating system: {system}")

    # --------------------------------------------------
    # Current SSID query
    # --------------------------------------------------
    def get_current_network(self) -> Optional[str]:  # noqa: C901 – complex but clear
        try:
            if self.os_type == "windows":
                result = subprocess.run(
                    ["netsh", "wlan", "show", "interfaces"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        line = line.strip()
                        if line.lower().startswith("ssid") and "bssid" not in line.lower():
                            return line.split(":", 1)[1].strip() or None
            elif self.os_type == "macos":
                # Try *airport* first
                result = subprocess.run(
                    [
                        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
                        "-I",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if " SSID:" in line:
                            return line.split("SSID:")[1].strip() or None
                # Fallback
                result = subprocess.run(
                    ["networksetup", "-getairportnetwork", "en0"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and "Current Wi-Fi Network:" in result.stdout:
                    return result.stdout.split("Current Wi-Fi Network:")[1].strip()
            else:  # linux / raspberry_pi
                result = subprocess.run(
                    ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if line.startswith("yes:"):
                            return line.split(":", 1)[1]
                # iwgetid fallback
                result = subprocess.run(["iwgetid", "-r"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return result.stdout.strip() or None
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Could not get current network: {exc}")
        return None

    # --------------------------------------------------
    # Connect helpers
    # --------------------------------------------------
    def connect_to_network(self, ssid: str, password: str, interface: Optional[str] = None) -> bool:
        logger.info(f"Connecting to network: {ssid}")
        try:
            if self.os_type == "windows":
                return self._connect_windows(ssid, password, interface)
            if self.os_type == "macos":
                return self._connect_macos(ssid, password, interface)
            return self._connect_linux(ssid, password, interface)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to connect to {ssid}: {exc}")
            return False

    # ---------------- macOS ----------------
    def _connect_macos(self, ssid: str, password: str, interface: Optional[str] = None) -> bool:
        interface = interface or "en0"
        try:
            result = subprocess.run(
                ["networksetup", "-setairportnetwork", interface, ssid, password],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for _ in range(10):
                    time.sleep(2)
                    if self.get_current_network() == ssid:
                        logger.info(f"Successfully connected to {ssid}")
                        return True
            logger.error(f"Failed to connect to {ssid}: {result.stderr.strip()}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Connection attempt timed out (macOS)")
            return False

    # ---------------- Linux / RPi ----------------
    def _connect_linux(self, ssid: str, password: str, interface: Optional[str] = None) -> bool:
        if not self._has_command("nmcli"):
            logger.error("NetworkManager (nmcli) not available")
            return False
        cmd = ["nmcli", "device", "wifi", "connect", ssid, "password", password]
        if interface:
            cmd.extend(["ifname", interface])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            time.sleep(3)
            if self.get_current_network() == ssid:
                logger.info(f"Successfully connected to {ssid}")
                return True
        logger.error(f"NetworkManager connection failed: {result.stderr.strip()}")
        return False

    # ---------------- Windows ----------------
    def _connect_windows(self, ssid: str, password: str, interface: Optional[str] = None) -> bool:
        interface = interface or "Wi-Fi"
        profile_xml_path = os.path.abspath(f"{ssid}_profile.xml")
        try:
            profile_xml = f"""<?xml version=\"1.0\"?>
<WLANProfile xmlns=\"http://www.microsoft.com/networking/WLAN/profile/v1\">
    <name>{ssid}</name>
    <SSIDConfig><SSID><name>{ssid}</name></SSID></SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{password}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>"""
            with open(profile_xml_path, "w", encoding="utf-8") as fp:
                fp.write(profile_xml)
            subprocess.run(
                [
                    "netsh",
                    "wlan",
                    "add",
                    "profile",
                    f"filename={profile_xml_path}",
                    f"interface={interface}",
                    "user=current",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            result = subprocess.run(
                [
                    "netsh",
                    "wlan",
                    "connect",
                    f"name={ssid}",
                    f"ssid={ssid}",
                    f"interface={interface}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for _ in range(10):
                    time.sleep(2)
                    if self.get_current_network() == ssid:
                        logger.info(f"Successfully connected to {ssid} (Windows)")
                        return True
            logger.error(f"Windows connection failed: {result.stderr.strip()}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Connection attempt timed out (Windows)")
            return False
        finally:
            try:
                os.remove(profile_xml_path)
            except OSError:
                pass

    # --------------------------------------------------
    def _has_command(self, cmd: str) -> bool:
        return subprocess.run(["which", cmd], capture_output=True).returncode == 0

# ------------------------------------------------------------
# Packet transmission helpers
# ------------------------------------------------------------

def _socket_tune(sock: socket.socket, buffer_size: int) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def send_packets_continuously(
    target_ip: str,
    packet_rate: int,
    stop_evt: threading.Event,
    packet_size: int = 64,
    buffer_size: int = 1_048_576,
) -> None:
    global packet_counter
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _socket_tune(sock, buffer_size)

    base_data = f"CSI_{datetime.now().strftime('%H%M%S')}_"
    payload = (base_data + "X" * max(0, packet_size - len(base_data)) ).encode()[:packet_size]

    logger.info(f"Starting HIGH-RATE transmission to {target_ip} at {packet_rate} pkt/s")
    interval = 0.0001 if packet_rate >= 10_000 else 1.0 / packet_rate
    next_time = time.time()

    while not stop_evt.is_set():
        try:
            sock.sendto(payload, (target_ip, 80))
            packet_counter += 1
            next_time += interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
            else:
                next_time = time.time()
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Packet send error: {exc}")
            time.sleep(0.0001)
    sock.close()
    logger.info(f"Packet transmission stopped. Sent {packet_counter} packets total.")


def log_serial_data(output_file: str, duration: int, skip_serial: bool = False) -> None:
    global csi_counter
    start_time = time.time()
    if skip_serial or not HAS_SERIAL:
        if not HAS_SERIAL:
            logger.warning("pyserial not installed; skipping serial logging")
        time.sleep(duration)
        return
    try:
        with serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01) as ser, open(output_file, "w") as f:
            f.write("timestamp,csi_data\n")
            while time.time() - start_time < duration and not stop_event.is_set():
                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if "CSI_DATA" in line:
                        csi_counter += 1
                        f.write(f"{datetime.now().isoformat(timespec='microseconds')},{line}\n")
                else:
                    time.sleep(0.001)
    except Exception as exc:
        logger.error(f"Serial logging error: {exc}")


def live_monitor() -> None:
    global packet_counter, csi_counter
    last_p, last_c = 0, 0
    while not stop_event.is_set():
        time.sleep(MONITOR_INTERVAL)
        logger.info(
            f"Live: {packet_counter - last_p} pkt/s | {csi_counter - last_c} CSI/s | "
            f"Total packets={packet_counter}, CSI={csi_counter}"
        )
        last_p, last_c = packet_counter, csi_counter

# ------------------------------------------------------------
# Active station orchestrator
# ------------------------------------------------------------
class ActiveStation:
    """High-level orchestrator that manages connection, optimisation and teardown."""

    def __init__(
        self,
        target_ssid: str,
        target_password: str,
        server_ip: str = "192.168.4.1",
        interface: Optional[str] = None,
        return_to_original: bool = True,
    ) -> None:
        self.target_ssid = target_ssid
        self.target_password = target_password
        self.server_ip = server_ip
        self.interface = interface or self._get_default_interface()
        self.return_to_original = return_to_original

        self.network_manager = NetworkManager()
        self.original_network: Optional[str] = None

    # --------------------------------------------------
    def _get_default_interface(self) -> str:
        sys_name = platform.system().lower()
        if sys_name.startswith("win"):
            return "Wi-Fi"
        if sys_name == "darwin":
            return "en0"
        return "wlan0"

    # --------------------------------------------------
    def connect_to_target_network(self) -> bool:
        if self.return_to_original:
            self.original_network = self.network_manager.get_current_network()
            if self.original_network:
                logger.info(f"Current network: {self.original_network} (will restore later)")
        if not self.network_manager.connect_to_network(self.target_ssid, self.target_password, self.interface):
            return False
        optimize_wifi(self.interface)
        return True

    # --------------------------------------------------
    def disconnect_and_restore(self) -> bool:
        if not self.return_to_original or not self.original_network:
            logger.info("Staying connected to target network")
            return True
        logger.info(f"Restoring connection to original network: {self.original_network}")
        try:
            if self.network_manager.os_type == "windows":
                result = subprocess.run(
                    [
                        "netsh",
                        "wlan",
                        "connect",
                        f"name={self.original_network}",
                        "interface=Wi-Fi",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    time.sleep(3)
                    if self.network_manager.get_current_network() == self.original_network:
                        logger.info("Successfully restored connection")
                        return True
            elif self.network_manager.os_type == "macos":
                result = subprocess.run(
                    ["networksetup", "-setairportnetwork", self.interface, self.original_network],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    time.sleep(3)
                    if self.network_manager.get_current_network() == self.original_network:
                        logger.info("Successfully restored connection")
                        return True
            else:  # linux / rpi
                result = subprocess.run(
                    ["nmcli", "connection", "up", self.original_network],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    time.sleep(3)
                    if self.network_manager.get_current_network() == self.original_network:
                        logger.info("Successfully restored connection")
                        return True
            logger.warning("Could not automatically restore the original connection")
            return False
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error during restoration: {exc}")
            return False

# ------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Active Station (Windows-enabled) for WiFi CSI collection")
    p.add_argument("--target-ssid", type=str, required=False, default="mywifi_ssid", help="Target WiFi SSID")
    p.add_argument("--target-password", type=str, required=False, default="mywifi_pass", help="Target WiFi password")
    p.add_argument("--server-ip", type=str, default="192.168.4.1", help="CSI collection AP IP")
    p.add_argument("--interface", type=str, default=None, help="WiFi interface name")
    p.add_argument("--rate", type=int, default=100, help="Packet rate (pkt/s)")
    p.add_argument("--duration", type=int, default=30, help="Test duration (s)")
    p.add_argument("--packet-size", type=int, default=64, help="UDP packet payload size (bytes)")
    p.add_argument("--buffer-size", type=int, default=1_048_576, help="Socket buffer size (bytes)")
    p.add_argument("--output", type=str, default="csi_output.csv", help="CSV file for captured CSI")
    p.add_argument("--no-serial", action="store_true", help="Skip serial logging even if serial is available")
    p.add_argument("--no-restore", action="store_true", help="Do not reconnect to the original AP")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    return p.parse_args()


def main() -> None:  # noqa: C901 – complex but unavoidable
    args = _parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    load_dotenv()
    target_ssid = args.target_ssid or os.getenv("TARGET_WIFI_SSID")
    target_password = args.target_password or os.getenv("TARGET_WIFI_PASSWORD")
    if not target_ssid or not target_password:
        logger.error("Target WiFi SSID and password are required")
        sys.exit(1)

    print(
        f"\nTarget Network: {target_ssid}\nPacket Rate: {args.rate} pkt/s\nPacket Size: {args.packet_size} bytes\n"
        f"Buffer Size: {args.buffer_size} bytes\nDuration: {args.duration} s\nOutput File: {args.output}\n"
        f"Restore Original: {'No' if args.no_restore else 'Yes'}"
    )

    station = ActiveStation(
        target_ssid=target_ssid,
        target_password=target_password,
        server_ip=args.server_ip,
        interface=args.interface,
        return_to_original=not args.no_restore,
    )

    try:
        if not station.connect_to_target_network():
            logger.error("Failed to connect to target network")
            sys.exit(1)

        threads = [
            threading.Thread(
                target=send_packets_continuously,
                args=(args.server_ip, args.rate, stop_event, args.packet_size, args.buffer_size),
            ),
            threading.Thread(target=log_serial_data, args=(args.output, args.duration, args.no_serial)),
            threading.Thread(target=live_monitor),
        ]
        logger.info("Starting CSI collection …")
        for t in threads:
            t.start()
        time.sleep(args.duration)
        stop_event.set()
        for t in threads:
            t.join()
        logger.info("CSI collection complete.")
        if not args.no_restore:
            station.disconnect_and_restore()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        stop_event.set()
    except Exception as exc:
        logger.error(f"Fatal error: {exc}")
        sys.exit(1)

# ------------------------------------------------------------
if __name__ == "__main__":
    main()

"""
Windows-Supported Active Station for WiFi CSI Collection with Live Monitoring
-----------------------------------------------------------------------------
This script extends the original `active_sta.py` implementation to add Windows
support while preserving *all* existing functionality for macOS, Linux and
Raspberry Pi.  It delegates the heavy-lifting (packet generation, serial
logging, live monitor, etc.) to the original module and only overrides the
network-management portions that differ on Windows.

Key Additions
-------------
1. `WindowsNetworkManager` – adds Windows detection plus connect / restore /
   status methods via `netsh wlan …`.
2. `ActiveStation` subclass – swaps in the new manager, provides Windows
   interface defaults ("Wi-Fi"), and handles restoration of the original
   network on Windows.

With these changes the CLI, arguments, logging behaviour and overall workflow
remain *identical* to the cross-platform script, so existing docs, automation
and integrations continue to work.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from typing import Optional

# Re-use everything else (packet sender, logger, etc.) from the original module
from . import active_sta as base  # relative import inside firmware/active_sta_py package

# -------------------- LOGGER --------------------
logger = base.logger  # use the same logger / log file

# -------------------- WINDOWS NETWORK MANAGER --------------------
class WindowsNetworkManager(base.NetworkManager):
    """Extend base NetworkManager with native Windows support (netsh)."""

    def _detect_os(self):
        system = platform.system().lower()
        if system.startswith("win"):
            return "windows"
        # fall back to the parent implementation for other systems
        return super()._detect_os()

    # ------------------------------------------------------------
    # Public helpers (override)
    # ------------------------------------------------------------
    def get_current_network(self) -> Optional[str]:
        if getattr(self, "os_type", None) != "windows":
            return super().get_current_network()

        try:
            result = subprocess.run(
                ["netsh", "wlan", "show", "interfaces"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    # Skip BSSID lines – we only need the SSID entry.
                    if line.lower().startswith("ssid") and "bssid" not in line.lower():
                        # Expected format: "SSID                   : MyWifi"
                        return line.split(":", 1)[1].strip() or None
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Could not get current network (Windows): {exc}")
        return None

    def connect_to_network(self, ssid: str, password: str, interface: Optional[str] = None) -> bool:
        """Connect to Wi-Fi on Windows using *netsh wlan* profile tricks."""
        if getattr(self, "os_type", None) != "windows":
            return super().connect_to_network(ssid, password, interface)
        return self._connect_windows(ssid, password, interface)

    # ------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------
    def _connect_windows(self, ssid: str, password: str, interface: Optional[str] = None) -> bool:
        interface = interface or "Wi-Fi"

        logger.info(f"[Windows] Connecting to network: {ssid} via interface {interface}")
        profile_xml_path = os.path.abspath(f"{ssid}_profile.xml")

        try:
            # 1. Generate a minimal WLAN profile XML.
            profile_xml = f"""<?xml version=\"1.0\"?>
<WLANProfile xmlns=\"http://www.microsoft.com/networking/WLAN/profile/v1\">
    <name>{ssid}</name>
    <SSIDConfig><SSID><name>{ssid}</name></SSID></SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{password}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>"""
            with open(profile_xml_path, "w", encoding="utf-8") as fp:
                fp.write(profile_xml)

            # 2. Add or overwrite the profile for the current user.
            subprocess.run(
                [
                    "netsh",
                    "wlan",
                    "add",
                    "profile",
                    f"filename={profile_xml_path}",
                    f"interface={interface}",
                    "user=current",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # 3. Attempt connection.
            result = subprocess.run(
                [
                    "netsh",
                    "wlan",
                    "connect",
                    f"name={ssid}",
                    f"ssid={ssid}",
                    f"interface={interface}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Wait (max ~20 s) for the connection to be established.
                for _ in range(10):
                    time.sleep(2)
                    if self.get_current_network() == ssid:
                        logger.info(f"Successfully connected to {ssid} (Windows)")
                        return True
            logger.error(f"Windows connection failed: {result.stderr.strip()}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Windows connection attempt timed out")
            return False
        except Exception as exc:  # pragma: no cover
            logger.error(f"Windows connection error: {exc}")
            return False
        finally:
            # Clean up temporary profile file.
            try:
                os.remove(profile_xml_path)
            except OSError:
                pass

# -------------------- ACTIVE STATION (WINDOWS AWARE) --------------------
class ActiveStation(base.ActiveStation):
    """ActiveStation drop-in replacement that swaps the network manager."""

    def __init__(
        self,
        target_ssid: str,
        target_password: str,
        server_ip: str = "192.168.4.1",
        interface: Optional[str] = None,
        return_to_original: bool = True,
    ) -> None:
        # Temporarily call parent with a dummy network manager; we'll overwrite below.
        super().__init__(
            target_ssid=target_ssid,
            target_password=target_password,
            server_ip=server_ip,
            interface=interface,
            return_to_original=return_to_original,
        )
        # Inject our Windows-capable manager.
        self.network_manager = WindowsNetworkManager()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def _get_default_interface(self):
        system = platform.system().lower()
        if system.startswith("win"):
            return "Wi-Fi"
        return super()._get_default_interface()

    def disconnect_and_restore(self):
        # Windows-specific restoration.
        if (
            self.network_manager.os_type == "windows"
            and self.return_to_original
            and self.original_network
        ):
            logger.info(
                f"Restoring connection to original network (Windows): {self.original_network}"
            )
            try:
                result = subprocess.run(
                    [
                        "netsh",
                        "wlan",
                        "connect",
                        f"name={self.original_network}",
                        "interface=Wi-Fi",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    time.sleep(3)
                    if self.network_manager.get_current_network() == self.original_network:
                        logger.info(
                            f"Successfully restored connection to {self.original_network}"
                        )
                        return True
            except Exception as exc:  # pragma: no cover
                logger.error(f"Error restoring original network on Windows: {exc}")

            logger.warning(f"Could not automatically restore {self.original_network}")
            return False
        # For non-Windows, fall back to parent behaviour.
        return super().disconnect_and_restore()

# -------------------- MAIN --------------------

def main():
    """CLI entrypoint – mirrors original script but instantiates new class."""

    parser = argparse.ArgumentParser(
        description="Active Station with added Windows support for WiFi CSI collection"
    )

    # Network arguments
    parser.add_argument(
        "--target-ssid",
        default="mywifi_ssid",
        type=str,
        required=False,
        help="SSID of target WiFi network",
    )
    parser.add_argument(
        "--target-password",
        default="mywifi_pass",
        type=str,
        required=False,
        help="Password for target WiFi network",
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="192.168.4.1",
        help="IP address of CSI collection AP",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=None,
        help="WiFi interface (auto-detected if not specified)",
    )

    # Transmission arguments (same as base)
    parser.add_argument("--rate", type=int, default=100, help="Packet transmission rate (pkt/s)")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--packet-size", type=int, default=64, help="Packet size in bytes")
    parser.add_argument(
        "--buffer-size", type=int, default=1048576, help="Socket buffer size in bytes"
    )

    # Output arguments
    parser.add_argument("--output", type=str, default="csi_output.csv", help="Output file for CSI data")
    parser.add_argument("--no-serial", action="store_true", help="Skip serial logging")

    # Control arguments
    parser.add_argument("--no-restore", action="store_true", help="Don't restore original network")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load env vars (re-use helper)
    base.load_dotenv()

    target_ssid = args.target_ssid or os.getenv("TARGET_WIFI_SSID")
    target_password = args.target_password or os.getenv("TARGET_WIFI_PASSWORD")

    if not target_ssid or not target_password:
        logger.error("Target WiFi SSID and password are required")
        sys.exit(1)

    print(f"\nTarget Network: {target_ssid}")
    print(f"Packet Rate: {args.rate} packets/second")
    print(f"Packet Size: {args.packet_size} bytes")
    print(f"Buffer Size: {args.buffer_size} bytes")
    print(f"Duration: {args.duration} seconds")
    print(f"Output File: {args.output}")
    print(f"Restore Original: {'No' if args.no_restore else 'Yes'}")

    station = ActiveStation(
        target_ssid=target_ssid,
        target_password=target_password,
        server_ip=args.server_ip,
        interface=args.interface,
        return_to_original=not args.no_restore,
    )

    try:
        # 1. Connect to target network.
        if not station.connect_to_target_network():
            logger.error("Failed to connect to target network")
            sys.exit(1)

        # 2. Start auxiliary threads (re-use base helpers).
        threads = [
            base.threading.Thread(
                target=base.send_packets_continuously,
                args=(
                    args.server_ip,
                    args.rate,
                    base.stop_event,
                    args.packet_size,
                    args.buffer_size,
                ),
            ),
            base.threading.Thread(
                target=base.log_serial_data,
                args=(args.output, args.duration, args.no_serial),
            ),
            base.threading.Thread(target=base.live_monitor),
        ]

        logger.info("Starting CSI collection (Windows-aware)…")
        for t in threads:
            t.start()

        # 3. Wait for completion.
        time.sleep(args.duration)
        base.stop_event.set()

        for t in threads:
            t.join()

        logger.info("CSI collection complete.")

        # 4. Restore original network.
        if not args.no_restore:
            station.disconnect_and_restore()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        base.stop_event.set()
    except Exception as exc:  # pragma: no cover
        logger.error(f"Fatal error: {exc}")
        sys.exit(1)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
