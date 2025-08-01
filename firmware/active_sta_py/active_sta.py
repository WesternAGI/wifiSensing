#!/usr/bin/env python3
"""
Cross-Platform Active Station for WiFi CSI Collection with Live Monitoring
This script connects to a WiFi network, optimizes connection for small UDP packets to maximize
CSI creation at the AP, provides live monitoring, and can restore the original network connection.

Supports: Linux, macOS, and Raspberry Pi
Features: High-rate packet transmission, serial CSI logging, live monitoring
"""

import os
import sys
import time
import socket
import argparse
import subprocess
import platform
import logging
import threading
from datetime import datetime

# Try to import optional dependencies
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# -------------------- CONFIG --------------------
WIFI_INTERFACE = "wlan0"
WIFI_CONNECT_TIMEOUT = 10
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
MAX_PACKET_RATE = 1000  # packets/sec
MONITOR_INTERVAL = 1.0  # seconds

# Global counters for live monitoring
packet_counter = 0
csi_counter = 0
stop_event = threading.Event()

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

# -------------------- UTILITIES --------------------
def run_cmd(cmd):
    """Execute shell command and return output."""
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        return ""

def optimize_wifi(interface):
    """Disable power save and increase performance for max CSI."""
    logger.info("Optimizing WiFi interface for max throughput...")
    try:
        os.system(f"sudo iw dev {interface} set power_save off 2>/dev/null")
        os.system(f"sudo iwconfig {interface} txpower auto 2>/dev/null")
        os.system(f"sudo ifconfig {interface} txqueuelen 2000 2>/dev/null")
        os.system(f"sudo iw {interface} set bitrates legacy-2.4 54 2>/dev/null")
    except Exception as e:
        logger.warning(f"WiFi optimization failed: {e}")

class NetworkManager:
    """Cross-platform network management for WiFi connections."""
    
    def __init__(self):
        self.os_type = self._detect_os()
        self.original_network = None
        logger.info(f"Detected OS: {self.os_type}")
    
    def _detect_os(self):
        """Detect the operating system type."""
        system = platform.system().lower()
        
        # Check for Raspberry Pi specifically
        if system == 'linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'raspberry pi' in cpuinfo or 'bcm' in cpuinfo:
                        return 'raspberry_pi'
            except:
                pass
            return 'linux'
        elif system == 'darwin':
            return 'macos'
        else:
            raise OSError(f"Unsupported operating system: {system}")
    
    def get_current_network(self):
        """Get the currently connected WiFi network SSID."""
        try:
            if self.os_type == 'macos':
                # Try airport command first
                result = subprocess.run([
                    '/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport',
                    '-I'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if ' SSID:' in line:
                            ssid = line.split('SSID:')[1].strip()
                            return ssid if ssid else None
                
                # Alternative method for macOS
                result = subprocess.run([
                    'networksetup', '-getairportnetwork', 'en0'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and 'Current Wi-Fi Network:' in result.stdout:
                    return result.stdout.split('Current Wi-Fi Network:')[1].strip()
                    
            elif self.os_type in ['linux', 'raspberry_pi']:
                # Try nmcli first (NetworkManager)
                result = subprocess.run([
                    'nmcli', '-t', '-f', 'active,ssid', 'dev', 'wifi'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('yes:'):
                            return line.split(':', 1)[1]
                
                # Try iwgetid as fallback
                result = subprocess.run([
                    'iwgetid', '-r'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    return result.stdout.strip()
                    
        except Exception as e:
            logger.warning(f"Could not get current network: {e}")
        
        return None
    
    def connect_to_network(self, ssid, password, interface=None):
        """Connect to a WiFi network."""
        logger.info(f"Connecting to network: {ssid}")
        
        try:
            if self.os_type == 'macos':
                return self._connect_macos(ssid, password, interface)
            elif self.os_type in ['linux', 'raspberry_pi']:
                return self._connect_linux(ssid, password, interface)
        except Exception as e:
            logger.error(f"Failed to connect to {ssid}: {e}")
            return False
    
    def _connect_macos(self, ssid, password, interface=None):
        """Connect to WiFi on macOS."""
        interface = interface or 'en0'
        
        try:
            # Connect to network
            result = subprocess.run([
                'networksetup', '-setairportnetwork', interface, ssid, password
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Wait for connection to establish
                for i in range(10):
                    time.sleep(2)
                    current = self.get_current_network()
                    if current == ssid:
                        logger.info(f"Successfully connected to {ssid}")
                        return True
                
            logger.error(f"Failed to connect to {ssid}: {result.stderr}")
            return False
            
        except subprocess.TimeoutExpired:
            logger.error("Connection attempt timed out")
            return False
        except Exception as e:
            logger.error(f"macOS connection error: {e}")
            return False
    
    def _connect_linux(self, ssid, password, interface=None):
        """Connect to WiFi on Linux/Raspberry Pi."""
        try:
            # Try NetworkManager first
            if self._has_command('nmcli'):
                return self._connect_networkmanager(ssid, password, interface)
            else:
                logger.error("NetworkManager (nmcli) not found")
                return False
                
        except Exception as e:
            logger.error(f"Linux connection error: {e}")
            return False
    
    def _connect_networkmanager(self, ssid, password, interface=None):
        """Connect using NetworkManager (nmcli)."""
        try:
            # Create new connection
            cmd = ['nmcli', 'device', 'wifi', 'connect', ssid, 'password', password]
            if interface:
                cmd.extend(['ifname', interface])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Verify connection
                time.sleep(3)
                current = self.get_current_network()
                if current == ssid:
                    logger.info(f"Successfully connected to {ssid}")
                    return True
            
            logger.error(f"NetworkManager connection failed: {result.stderr}")
            return False
            
        except subprocess.TimeoutExpired:
            logger.error("NetworkManager connection timed out")
            return False
    
    def _has_command(self, command):
        """Check if a command is available."""
        try:
            subprocess.run(['which', command], check=True, 
                         capture_output=True, timeout=5)
            return True
        except:
            return False

# -------------------- PACKET TRANSMISSION --------------------
def send_packets_continuously(target_ip, packet_rate, stop_event):
    """Send UDP packets continuously at specified rate."""
    global packet_counter
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # increase buffer
    
    # Create optimized packet
    base_data = f"CSI_PKT_{datetime.now().strftime('%H%M%S')}_"
    payload = (base_data + "A" * (120 - len(base_data))).encode('utf-8')

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
    """Log CSI data from serial port."""
    global csi_counter
    start_time = time.time()
    
    if skip_serial or not HAS_SERIAL:
        if not HAS_SERIAL:
            logger.warning("pyserial not installed, skipping serial logging")
        else:
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
    """Monitor packet and CSI rates in real-time."""
    global packet_counter, csi_counter
    last_packets = 0
    last_csi = 0
    
    while not stop_event.is_set():
        time.sleep(MONITOR_INTERVAL)
        pps = packet_counter - last_packets
        cps = csi_counter - last_csi
        logger.info(f"Live: {pps} pkt/s | {cps} CSI/s | Total packets={packet_counter}, CSI={csi_counter}")
        last_packets, last_csi = packet_counter, csi_counter

class ActiveStation:
    """Enhanced Active Station with cross-platform support and live monitoring."""
    
    def __init__(self, target_ssid, target_password, server_ip='192.168.4.1', 
                 interface=None, return_to_original=True):
        self.target_ssid = target_ssid
        self.target_password = target_password
        self.server_ip = server_ip
        self.interface = interface or self._get_default_interface()
        self.return_to_original = return_to_original
        
        # Initialize network manager
        self.network_manager = NetworkManager()
        self.original_network = None
        
    def _get_default_interface(self):
        """Get default WiFi interface based on OS."""
        system = platform.system().lower()
        if system == 'darwin':
            return 'en0'
        else:
            return 'wlan0'
    
    def connect_to_target_network(self):
        """Connect to the target WiFi network."""
        if self.return_to_original:
            self.original_network = self.network_manager.get_current_network()
            if self.original_network:
                logger.info(f"Current network: {self.original_network} (will restore later)")
        
        logger.info(f"Connecting to target network: {self.target_ssid}")
        success = self.network_manager.connect_to_network(
            self.target_ssid, self.target_password, self.interface
        )
        
        if success:
            optimize_wifi(self.interface)
            logger.info("Successfully connected and optimized for CSI collection")
            return True
        else:
            logger.error(f"Failed to connect to target network: {self.target_ssid}")
            return False
    
    def disconnect_and_restore(self):
        """Restore original network connection."""
        if not self.return_to_original or not self.original_network:
            logger.info("Staying connected to target network")
            return True
        
        logger.info(f"Restoring connection to original network: {self.original_network}")
        
        try:
            if self.network_manager.os_type == 'macos':
                result = subprocess.run([
                    'networksetup', '-setairportnetwork', 
                    self.interface, self.original_network
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    time.sleep(3)
                    current = self.network_manager.get_current_network()
                    if current == self.original_network:
                        logger.info(f"Successfully restored connection to {self.original_network}")
                        return True
                        
            elif self.network_manager.os_type in ['linux', 'raspberry_pi']:
                result = subprocess.run([
                    'nmcli', 'connection', 'up', self.original_network
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    time.sleep(3)
                    current = self.network_manager.get_current_network()
                    if current == self.original_network:
                        logger.info(f"Successfully restored connection to {self.original_network}")
                        return True
            
            logger.warning(f"Could not automatically restore {self.original_network}")
            return False
            
        except Exception as e:
            logger.error(f"Error restoring original network: {e}")
            return False

# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser(
        description='Cross-platform Active Station for WiFi CSI collection with live monitoring'
    )
    
    # Network arguments
    parser.add_argument("--target-ssid", default="mywifi_ssid", type=str, required=False,
                      help="SSID of target WiFi network")
    parser.add_argument("--target-password", default="mywifi_pass", type=str, required=False,
                      help="Password for target WiFi network")
    parser.add_argument("--server-ip", type=str, default="192.168.4.1",
                      help="IP address of CSI collection AP")
    parser.add_argument("--interface", type=str, default=None,
                      help="WiFi interface (auto-detected if not specified)")
    
    # Transmission arguments
    parser.add_argument("--rate", type=int, default=100,
                      help="Packet transmission rate (pkt/s)")
    parser.add_argument("--duration", type=int, default=30,
                      help="Duration in seconds")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="csi_output.csv",
                      help="Output file for CSI data")
    parser.add_argument("--no-serial", action="store_true",
                      help="Skip serial logging")
    
    # Control arguments
    parser.add_argument("--no-restore", action="store_true",
                      help="Don't restore original network")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    target_ssid = args.target_ssid or os.getenv('TARGET_WIFI_SSID')
    target_password = args.target_password or os.getenv('TARGET_WIFI_PASSWORD')
    
    if not target_ssid or not target_password:
        logger.error("Target WiFi SSID and password are required")
        sys.exit(1)
    
    print(f"\nTarget Network: {target_ssid}")
    print(f"Packet Rate: {args.rate} packets/second")
    print(f"Duration: {args.duration} seconds")
    print(f"Output File: {args.output}")
    print(f"Restore Original: {'No' if args.no_restore else 'Yes'}")
    
    # Create active station
    station = ActiveStation(
        target_ssid=target_ssid,
        target_password=target_password,
        server_ip=args.server_ip,
        interface=args.interface,
        return_to_original=not args.no_restore
    )
    
    try:
        # Connect to target network
        if not station.connect_to_target_network():
            logger.error("Failed to connect to target network")
            sys.exit(1)
        
        # Start threads for packet transmission, CSI logging, and monitoring
        threads = [
            threading.Thread(target=send_packets_continuously, 
                           args=(args.server_ip, args.rate, stop_event)),
            threading.Thread(target=log_serial_data, 
                           args=(args.output, args.duration, args.no_serial)),
            threading.Thread(target=live_monitor)
        ]
        
        logger.info("Starting CSI collection...")
        for t in threads:
            t.start()
        
        # Wait for completion
        time.sleep(args.duration)
        stop_event.set()
        
        for t in threads:
            t.join()
        
        logger.info("CSI collection complete.")
        
        # Restore original network
        if not args.no_restore:
            station.disconnect_and_restore()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        stop_event.set()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
