#!/usr/bin/env python3
"""
Active Station (STA) implementation in Python for Linux/macOS.
This script connects to a WiFi network and sends data to trigger CSI collection on the access point.
"""

import os
import sys
import time
import argparse
import subprocess
import logging
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
logger = logging.getLogger('active_sta')

class ActiveStation:
    def __init__(self, ssid, password, server_ip='192.168.4.1', server_port=80, channel=1, interface=None):
        """
        Initialize the Active Station.
        
        Args:
            ssid (str): SSID of the target WiFi network
            password (str): Password for the WiFi network
            server_ip (str): IP address of the server (AP) to send data to
            server_port (int): Port of the server (AP) to send data to
            channel (int): WiFi channel to use (default: 1)
            interface (str, optional): Network interface to use (e.g., 'en0' for macOS, 'wlan0' for Linux)
        """
        self.ssid = ssid
        self.password = password
        self.server_ip = server_ip
        self.server_port = server_port
        self.channel = channel
        self.interface = interface if interface else self._get_wifi_interface()
        self.connected = False
        self.debug = False
        
        # Set up socket for communication with AP
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(2)  # 2 second timeout
        
    def _get_wifi_interface(self):
        """Get the WiFi interface name based on the OS."""
        system = platform.system().lower()
        
        if system == 'linux':
            # Common WiFi interface names on Linux
            possible_interfaces = ['wlan0', 'wlp2s0', 'wlp3s0', 'wlan1', 'wlo1', 'wlp0s20f3']
            
            # Check which interface exists
            for interface in possible_interfaces:
                if os.path.exists(f'/sys/class/net/{interface}'):
                    logger.debug(f"Found Linux WiFi interface: {interface}")
                    return interface
                    
        elif system == 'darwin':  # macOS
            try:
                # Use networksetup to find the active WiFi interface
                result = subprocess.run(
                    ['networksetup', '-listallhardwareports'],
                    capture_output=True, text=True, check=True
                )
                
                # Parse the output to find the WiFi interface
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if 'Wi-Fi' in line or 'AirPort' in line:
                        # The next line should contain the device name
                        if i + 1 < len(lines):
                            device_line = lines[i + 1]
                            if 'Device' in device_line:
                                interface = device_line.split(':')[-1].strip()
                                logger.debug(f"Found macOS WiFi interface: {interface}")
                                return interface
                
                # Fallback to common interface names if parsing fails
                possible_interfaces = ['en0', 'en1']
                for interface in possible_interfaces:
                    try:
                        # Check if the interface is a WiFi interface
                        result = subprocess.run(
                            ['ifconfig', interface],
                            capture_output=True, text=True
                        )
                        if 'status: active' in result.stdout or 'UP' in result.stdout:
                            logger.debug(f"Found active interface via ifconfig: {interface}")
                            return interface
                    except:
                        continue
                        
            except Exception as e:
                logger.warning(f"Error detecting WiFi interface: {e}")
                # Fall through to the error below
        else:
            raise OSError(f"Unsupported operating system: {system}")
            
        # If we get here, we couldn't determine the interface
        available_interfaces = []
        try:
            if system == 'linux':
                available_interfaces = os.listdir('/sys/class/net/')
            elif system == 'darwin':
                result = subprocess.run(['ifconfig', '-l'], capture_output=True, text=True)
                available_interfaces = result.stdout.strip().split()
        except:
            pass
            
        error_msg = (
            "Could not determine WiFi interface. "
            f"Available interfaces: {available_interfaces}. "
            "Please specify the interface manually using the --interface argument."
        )
        raise OSError(error_msg)
    
    def connect_to_wifi(self):
        """Skip WiFi connection check and proceed."""
        logger.info("Skipping WiFi connection check. Please ensure you're connected to the ESP32's WiFi network.")
        logger.info(f"Target AP: {self.server_ip}:{self.server_port}")
        self.connected = True
        return True
    
    def send_data(self, data=None):
        """
        Send data to the server (AP).
        
        Args:
            data (bytes, optional): Data to send. If None, a default packet will be created.
            
        Returns:
            bool: True if data was sent successfully, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to WiFi")
            return False
            
        try:
            # Create a more noticeable packet if none provided
            if data is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                packet = f"CSI_DATA_PACKET_{timestamp}".encode('utf-8')
            else:
                next_time = time.time()
        except Exception as e:
            print('F', end='', flush=True)  # F for Failure
            logger.error(f"Failed to send data to {self.server_ip}:{self.server_port}: {e}")
            return False
    
    def run(self, interval=1.0, duration=None):
        """
        Run the active station.
        
        Args:
            interval (float): Time in seconds between sending data packets (default: 1.0)
            duration (int): Duration in seconds to run for (default: None, run indefinitely)
        """
        print("\n" + "="*50)
        print(f"Starting Active Station")
        print(f"Interface: {self.interface}" if self.interface else "Interface: Auto-detected")
        print(f"Target AP: {self.server_ip}:{self.server_port}")
        print(f"Channel: {self.channel}")
        print(f"Interval: {interval} seconds")
        print("="*50 + "\n")
        
        # Skip WiFi connection check
        self.connected = True

        # If interval is very small, warn user
        if interval < 0.01:
            print("Warning: Interval is very small. System/network may not keep up.")
        
        print("\n" + "="*50)
        print("Sending data packets:")
        print("  . = Packet sent successfully")
        print("  T = Timeout")
        print("  E = Address error")
        print("  F = Other failure")
        print("\nPress Ctrl+C to stop")
        print("-"*50 + "\n")
        
        start_time = time.time()
        packet_count = 0
        last_log_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while True:
                current_time = time.time()
                
                # Check if we've exceeded the duration
                if duration and (current_time - start_time) > duration:
                    print("\n" + "="*50)
                    print(f"Reached duration limit of {duration} seconds. Stopping.")
                    break
                
                # Log status every 5 seconds
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    success_rate = (packet_count / (elapsed / interval)) * 100 if elapsed > 0 else 0
                    print(f"\n[Status] Running for {elapsed:.1f}s | "
                          f"Packets: {packet_count} | "
                          f"Rate: {packet_count/max(elapsed, 0.1):.1f} pkt/s | "
                          f"Success: {success_rate:.1f}%")
                    last_log_time = current_time
                
                # Send data packet
                if self.send_data():
                    packet_count += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\n\nToo many failures ({consecutive_failures}). Check your connection to the ESP32.")
                        print(f"- Make sure you're connected to the ESP32's WiFi network")
                        print(f"- Verify the ESP32 is running in AP mode with IP {self.server_ip}")
                        print(f"- Check if any firewall is blocking the connection\n")
                        consecutive_failures = 0  # Reset counter after showing message
                
                # Calculate sleep time to maintain consistent interval
                elapsed_in_loop = time.time() - current_time
                sleep_time = max(0, interval - elapsed_in_loop)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n" + "="*50)
            print("Received keyboard interrupt. Stopping...")
            
        except Exception as e:
            print("\n" + "="*50)
            logger.error(f"Unexpected error: {e}", exc_info=True)
            
        finally:
            elapsed = time.time() - start_time
            print("\n" + "="*50)
            print("Active Station Stopped")
            print(f"Total runtime: {elapsed:.1f} seconds")
            print(f"Total packets sent: {packet_count}")
            if elapsed > 0:
                print(f"Average packet rate: {packet_count/elapsed:.2f} packets/second")
                print(f"Success rate: {(packet_count / (elapsed / interval) * 100):.1f}%")
            print("="*50 + "\n")
            
            try:
                self.socket.close()
            except Exception as e:
                if self.debug:
                    logger.debug(f"Error closing socket: {e}")


def main():
    # Parse command line arguments
    #python active_sta.py --ssid mywifi_ssid --password mywifi_pass --debug
    parser = argparse.ArgumentParser(
        description='Active Station for WiFi CSI collection\n\n'
        'IMPORTANT: Before running this script, please ensure:\n'
        '1. Your ESP32 is powered on and in Access Point (AP) mode\n'
        '2. Your computer is connected to the ESP32\'s WiFi network\n'
        '3. The ESP32 IP is 192.168.4.1 (default for ESP32 in AP mode)\n\n',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--ssid', type=str, default='mywifi_ssid',
                      help='SSID of the target WiFi network (default: mywifi_ssid)')
    parser.add_argument('--password', type=str, default='mywifi_pass',
                      help='Password for the WiFi network (default: mywifi_pass)')
    parser.add_argument('--server-ip', type=str, default='192.168.4.1',
                      help='IP address of the ESP32 (default: 192.168.4.1)')
    parser.add_argument('--server-port', type=int, default=80,
                      help='Port of the ESP32 (default: 80)')
    parser.add_argument('--channel', type=int, default=1,
                      help='WiFi channel (default: 1)')
    parser.add_argument('--interface', type=str, default=None,
                      help='Network interface (e.g., en0, wlan0). Auto-detected if not specified')
    parser.add_argument('--interval', type=float, default=1.0,
                      help='Time between packets in seconds (default: 1.0)')
    parser.add_argument('--duration', type=int, default=None,
                      help='Run duration in seconds (default: run indefinitely)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Print connection help
    print("\n" + "="*70)
    print(f"Targeting ESP32 at {args.server_ip}:{args.server_port}")
    print("="*70)
    
    if args.interface:
        print(f"Using network interface: {args.interface}")
    else:
        print("Auto-detecting network interface...")
    
    print("\nMake sure your computer is connected to the ESP32's WiFi network:")
    print(f"  SSID:     {args.ssid}")
    print(f"  Password: {'*' * len(args.password)}")
    print("\nIf you see many failures (T/E/F), please check:")
    print("  1. Your computer is connected to the ESP32's WiFi")
    print("  2. The ESP32 is powered on and in AP mode")
    print("  3. No firewall is blocking the connection")
    print("="*70 + "\n")
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Get SSID and password from arguments, environment variables, or prompt
    ssid = args.ssid or os.getenv('WIFI_SSID')
    password = args.password or os.getenv('WIFI_PASSWORD')
    
    if not ssid:
        ssid = input("Enter WiFi SSID: ")
    if not password:
        import getpass
        password = getpass.getpass("Enter WiFi password: ")
    
    # Create and run the active station
    station = ActiveStation(
        ssid=ssid,
        password=password,
        server_ip=args.server_ip,
        server_port=args.server_port,
        channel=args.channel,
        interface=args.interface
    )
    
    # Store debug flag
    station.debug = args.debug
    
    station.run(interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    main()

# (venv) (base) lab7@lab7-XPS-8940:~/Desktop/wifiSensing/firmware/active_sta_py$ python active_sta.py --ssid mywifi_ssid --password mywifi_pass --debug
# 2025-07-13 22:39:50,317 - active_sta - DEBUG - Found Linux WiFi interface: wlo1

# ==================================================
# Starting Active Station
# Interface: wlo1
# Connecting to SSID: mywifi_ssid
# Target AP: 192.168.4.1:80
# Channel: 1
# Interval: 1.0 seconds
# ==================================================

# Error: Connection activation failed: Secrets were required, but not provided.
# 2025-07-13 22:40:35,695 - active_sta - ERROR - Failed to connect to WiFi: Command '['nmcli', 'device', 'wifi', 'connect', 'mywifi_ssid', 'password', 'mywifi_pass']' returned non-zero exit status 4.
# 2025-07-13 22:40:35,695 - active_sta - ERROR - Failed to connect to WiFi. Exiting.
# (venv) (base) lab7@lab7-XPS-8940:~/Desktop/wifiSensing/firmware/active_sta_py$ ^C
# (venv) (base) lab7@lab7-XPS-8940:~/Desktop/wifiSensing/firmware/active_sta_py$ 

