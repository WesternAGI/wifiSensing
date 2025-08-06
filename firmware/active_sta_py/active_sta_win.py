#!/usr/bin/env python3
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
