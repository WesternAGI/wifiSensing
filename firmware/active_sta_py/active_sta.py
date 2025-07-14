#!/usr/bin/env python3
"""
Active Station (STA) implementation in Python for Linux/macOS.
This script connects to a WiFi network and sends data to trigger CSI collection on the access point.
"""

import os
import time
import socket
import argparse
import subprocess
import platform
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('active_sta.log')
    ]
)
logger = logging.getLogger('active_sta')

class ActiveStation:
    def __init__(self, ssid, password, server_ip='192.168.4.1', server_port=80, channel=1):
        """
        Initialize the Active Station.
        
        Args:
            ssid (str): SSID of the target WiFi network
            password (str): Password for the WiFi network
            server_ip (str): IP address of the server (AP) to send data to
            server_port (int): Port of the server (AP) to send data to
            channel (int): WiFi channel to use (default: 1)
        """
        self.ssid = ssid
        self.password = password
        self.server_ip = server_ip
        self.server_port = server_port
        self.channel = channel
        self.interface = self._get_wifi_interface()
        self.connected = False
        
        # Set up socket for communication with AP
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(5)  # 5 second timeout
        
    def _get_wifi_interface(self):
        """Get the WiFi interface name based on the OS."""
        system = platform.system().lower()
        if system == 'linux':
            # Common WiFi interface names on Linux
            possible_interfaces = ['wlan0', 'wlp2s0', 'wlp3s0', 'wlan1']
        elif system == 'darwin':  # macOS
            possible_interfaces = ['en0', 'en1']
        else:
            raise OSError("Unsupported operating system")
            
        # Check which interface exists
        for interface in possible_interfaces:
            if os.path.exists(f'/sys/class/net/{interface}'):
                return interface
                
        raise OSError("Could not determine WiFi interface")
    
    def connect_to_wifi(self):
        """Connect to the WiFi network."""
        system = platform.system().lower()
        
        try:
            if system == 'linux':
                # Using nmcli to connect to WiFi on Linux
                subprocess.run([
                    'nmcli', 'device', 'wifi', 'connect', 
                    self.ssid, 'password', self.password
                ], check=True)
                
            elif system == 'darwin':  # macOS
                # Using networksetup to connect to WiFi on macOS
                subprocess.run([
                    'networksetup', '-setairportnetwork', self.interface,
                    self.ssid, self.password
                ], check=True)
                
            logger.info(f"Successfully connected to {self.ssid}")
            self.connected = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to connect to WiFi: {e}")
            self.connected = False
            return False
    
    def send_data(self, data=b"CSI_DATA_PACKET"):
        """
        Send data to the server (AP).
        
        Args:
            data (bytes): Data to send (default: b"CSI_DATA_PACKET")
            
        Returns:
            bool: True if data was sent successfully, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to WiFi")
            return False
            
        try:
            self.socket.sendto(data, (self.server_ip, self.server_port))
            logger.debug(f"Sent {len(data)} bytes to {self.server_ip}:{self.server_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            return False
    
    def run(self, interval=1.0, duration=None):
        """
        Run the active station.
        
        Args:
            interval (float): Time in seconds between sending data packets (default: 1.0)
            duration (int): Duration in seconds to run for (default: None, run indefinitely)
        """
        logger.info(f"Starting Active Station on interface {self.interface}")
        logger.info(f"Connecting to {self.ssid}...")
        
        if not self.connect_to_wifi():
            logger.error("Failed to connect to WiFi. Exiting.")
            return
            
        logger.info(f"Sending data to {self.server_ip}:{self.server_port} every {interval} seconds")
        
        start_time = time.time()
        packet_count = 0
        
        try:
            while True:
                # Check if we've exceeded the duration
                if duration and (time.time() - start_time) > duration:
                    logger.info(f"Reached duration limit of {duration} seconds. Stopping.")
                    break
                
                # Send data packet
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                data = f"CSI_DATA_PACKET_{timestamp}".encode('utf-8')
                
                if self.send_data(data):
                    packet_count += 1
                    if packet_count % 10 == 0:  # Log every 10 packets
                        logger.info(f"Sent {packet_count} packets")
                
                # Wait for the next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("\nReceived keyboard interrupt. Stopping...")
            
        finally:
            logger.info(f"Stopped. Sent a total of {packet_count} packets.")
            self.socket.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Active Station for WiFi CSI collection')
    parser.add_argument('--ssid', type=str, help='SSID of the target WiFi network')
    parser.add_argument('--password', type=str, help='Password for the WiFi network')
    parser.add_argument('--server-ip', type=str, default='192.168.4.1',
                       help='IP address of the server (AP) to send data to (default: 192.168.4.1)')
    parser.add_argument('--server-port', type=int, default=80,
                       help='Port of the server (AP) to send data to (default: 80)')
    parser.add_argument('--channel', type=int, default=1,
                       help='WiFi channel to use (default: 1)')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Time in seconds between sending data packets (default: 1.0)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in seconds to run for (default: run indefinitely)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
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
        channel=args.channel
    )
    
    station.run(interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    main()
