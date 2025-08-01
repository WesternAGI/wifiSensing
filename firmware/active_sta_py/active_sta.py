#!/usr/bin/env python3
"""
Active Station (STA) implementation in Python for cross-platform WiFi CSI collection.
This script connects to a WiFi network, optimizes connection for small UDP packets to maximize
CSI creation at the AP, then disconnects and reconnects to the original network.

Supports: Linux, macOS, and Raspberry Pi
"""

import os
import sys
import time
import socket
import argparse
import subprocess
import platform
import logging
import json
import re
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
            # Turn WiFi off and on to ensure clean connection
            subprocess.run(['networksetup', '-setairportpower', interface, 'off'], 
                         check=True, timeout=10)
            time.sleep(2)
            subprocess.run(['networksetup', '-setairportpower', interface, 'on'], 
                         check=True, timeout=10)
            time.sleep(3)
            
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
            # Fallback to wpa_supplicant
            elif self._has_command('wpa_supplicant'):
                return self._connect_wpa_supplicant(ssid, password, interface)
            else:
                logger.error("No supported WiFi management tool found (nmcli or wpa_supplicant)")
                return False
                
        except Exception as e:
            logger.error(f"Linux connection error: {e}")
            return False
    
    def _connect_networkmanager(self, ssid, password, interface=None):
        """Connect using NetworkManager (nmcli)."""
        try:
            # Check if connection profile exists
            result = subprocess.run([
                'nmcli', 'connection', 'show', ssid
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Profile exists, just activate it
                result = subprocess.run([
                    'nmcli', 'connection', 'up', ssid
                ], capture_output=True, text=True, timeout=30)
            else:
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
    
    def _connect_wpa_supplicant(self, ssid, password, interface=None):
        """Connect using wpa_supplicant (fallback method)."""
        interface = interface or 'wlan0'
        
        try:
            # Create temporary wpa_supplicant config
            config_content = f"""
network={{
    ssid="{ssid}"
    psk="{password}"
    key_mgmt=WPA-PSK
}}
"""
            config_path = '/tmp/wpa_temp.conf'
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # Kill existing wpa_supplicant processes
            subprocess.run(['sudo', 'killall', 'wpa_supplicant'], 
                         capture_output=True, timeout=5)
            time.sleep(1)
            
            # Start wpa_supplicant
            subprocess.run([
                'sudo', 'wpa_supplicant', '-B', '-i', interface, 
                '-c', config_path, '-D', 'wext'
            ], check=True, timeout=10)
            
            # Get IP address
            subprocess.run(['sudo', 'dhclient', interface], 
                         check=True, timeout=20)
            
            # Verify connection
            time.sleep(3)
            current = self.get_current_network()
            if current == ssid:
                logger.info(f"Successfully connected to {ssid}")
                os.remove(config_path)
                return True
            
            os.remove(config_path)
            return False
            
        except Exception as e:
            logger.error(f"wpa_supplicant connection failed: {e}")
            if os.path.exists('/tmp/wpa_temp.conf'):
                os.remove('/tmp/wpa_temp.conf')
            return False
    
    def _has_command(self, command):
        """Check if a command is available."""
        try:
            subprocess.run(['which', command], check=True, 
                         capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def optimize_for_small_packets(self, interface=None):
        """Optimize network settings for small UDP packet transmission."""
        logger.info("Optimizing network settings for small UDP packets...")
        
        try:
            if self.os_type == 'macos':
                self._optimize_macos(interface)
            elif self.os_type in ['linux', 'raspberry_pi']:
                self._optimize_linux(interface)
        except Exception as e:
            logger.warning(f"Could not optimize network settings: {e}")
    
    def _optimize_macos(self, interface=None):
        """Optimize network settings on macOS."""
        interface = interface or 'en0'
        
        try:
            # Set socket buffer sizes (requires root)
            commands = [
                ['sudo', 'sysctl', '-w', 'net.inet.udp.sendspace=65536'],
                ['sudo', 'sysctl', '-w', 'net.inet.udp.recvspace=65536'],
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd, check=True, timeout=5, capture_output=True)
                except:
                    pass  # Continue even if some optimizations fail
                    
        except Exception as e:
            logger.debug(f"macOS optimization warning: {e}")
    
    def _optimize_linux(self, interface=None):
        """Optimize network settings on Linux/Raspberry Pi."""
        interface = interface or 'wlan0'
        
        try:
            # Optimize socket buffers and network parameters
            commands = [
                ['sudo', 'sysctl', '-w', 'net.core.rmem_max=134217728'],
                ['sudo', 'sysctl', '-w', 'net.core.wmem_max=134217728'],
                ['sudo', 'sysctl', '-w', 'net.core.rmem_default=65536'],
                ['sudo', 'sysctl', '-w', 'net.core.wmem_default=65536'],
                ['sudo', 'sysctl', '-w', 'net.ipv4.udp_mem=102400 873800 16777216'],
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd, check=True, timeout=5, capture_output=True)
                except:
                    pass  # Continue even if some optimizations fail
                    
        except Exception as e:
            logger.debug(f"Linux optimization warning: {e}")

class ActiveStation:
    def __init__(self, target_ssid, target_password, server_ip='192.168.4.1', 
                 server_port=80, channel=1, interface=None, return_to_original=True):
        """
        Initialize the Active Station.
        
        Args:
            target_ssid (str): SSID of the target WiFi network
            target_password (str): Password for the target WiFi network
            server_ip (str): IP address of the server (AP) to send data to
            server_port (int): Port of the server (AP) to send data to
            channel (int): WiFi channel to use (default: 1)
            interface (str, optional): Network interface to use
            return_to_original (bool): Whether to return to original network after completion
        """
        self.target_ssid = target_ssid
        self.target_password = target_password
        self.server_ip = server_ip
        self.server_port = server_port
        self.channel = channel
        self.interface = interface if interface else self._get_wifi_interface()
        self.return_to_original = return_to_original
        self.connected = False
        self.debug = False
        
        # Initialize network manager
        self.network_manager = NetworkManager()
        self.original_network = None
        
        # Set up socket for communication with AP
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(2)  # 2 second timeout
        
        # Optimize socket for small packets
        try:
            # Set socket buffer sizes for optimal small packet performance
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
            # Enable broadcast if needed
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception as e:
            logger.warning(f"Could not optimize socket: {e}")

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
    
    def connect_to_target_network(self):
        """Connect to the target WiFi network for CSI data collection."""
        # Store original network for later restoration
        if self.return_to_original:
            self.original_network = self.network_manager.get_current_network()
            if self.original_network:
                logger.info(f"Current network: {self.original_network} (will restore later)")
        
        # Connect to target network
        logger.info(f"Connecting to target network: {self.target_ssid}")
        success = self.network_manager.connect_to_network(
            self.target_ssid, self.target_password, self.interface
        )
        
        if success:
            # Optimize network settings for small packet transmission
            self.network_manager.optimize_for_small_packets(self.interface)
            self.connected = True
            logger.info("Successfully connected and optimized for CSI collection")
            return True
        else:
            logger.error(f"Failed to connect to target network: {self.target_ssid}")
            return False
    
    def disconnect_and_restore(self):
        """Disconnect from target network and restore original connection."""
        if not self.return_to_original or not self.original_network:
            logger.info("Staying connected to target network")
            return True
        
        logger.info(f"Restoring connection to original network: {self.original_network}")
        
        # For the original network, we don't have the password, so we try to reconnect
        # using existing profiles/keychains
        try:
            if self.network_manager.os_type == 'macos':
                # On macOS, try to connect without password (using keychain)
                result = subprocess.run([
                    'networksetup', '-setairportnetwork', 
                    self.interface or 'en0', self.original_network
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    time.sleep(3)
                    current = self.network_manager.get_current_network()
                    if current == self.original_network:
                        logger.info(f"Successfully restored connection to {self.original_network}")
                        return True
                        
            elif self.network_manager.os_type in ['linux', 'raspberry_pi']:
                # On Linux, try to activate existing connection profile
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
            logger.info("Please manually reconnect to your original network")
            return False
            
        except Exception as e:
            logger.error(f"Error restoring original network: {e}")
            return False

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
                packet = data

            # Set a shorter timeout for the socket
            self.socket.settimeout(0.1)  # 100 ms timeout for faster retries

            # Send the packet
            self.socket.sendto(packet, (self.server_ip, self.server_port))

            # Print a dot for each successful send
            print('.', end='', flush=True)

            if self.debug:
                logger.debug(f"Sent {len(packet)} bytes to {self.server_ip}:{self.server_port}")
                logger.debug(f"Packet content: {packet}")

            return True
            
        except socket.timeout:
            print('T', end='', flush=True)  # T for Timeout
            logger.warning(f"Timeout while sending to {self.server_ip}:{self.server_port}")
            return False
            
        except socket.gaierror as e:
            print('E', end='', flush=True)  # E for Error
            logger.error(f"Address-related error: {e}")
            return False
            
        except Exception as e:
            print('F', end='', flush=True)  # F for Failure
            logger.error(f"Failed to send data to {self.server_ip}:{self.server_port}: {e}")
            return False
    
    def run(self, interval=0.01, duration=60, packet_size=64):
        """
        Run the active station with optimized small packet transmission.
        
        Args:
            interval (float): Time in seconds between packets (default: 0.01 for 100 pps)
            duration (int): Duration in seconds to run for (default: 60)
            packet_size (int): Size of UDP packets in bytes (default: 64)
        """
        print("\n" + "="*60)
        print(f"Active Station for WiFi CSI Collection")
        print(f"OS: {self.network_manager.os_type.replace('_', ' ').title()}")
        print(f"Interface: {self.interface}")
        print(f"Target Network: {self.target_ssid}")
        print(f"Target AP: {self.server_ip}:{self.server_port}")
        print(f"Packet Rate: {1/interval:.1f} packets/second")
        print(f"Packet Size: {packet_size} bytes")
        print(f"Duration: {duration} seconds")
        print("="*60 + "\n")
        
        # Connect to target network
        if not self.connect_to_target_network():
            logger.error("Failed to connect to target network. Exiting.")
            return False
        
        # Create optimized packet data
        base_data = f"CSI_PKT_{datetime.now().strftime('%H%M%S')}_"
        padding_size = max(0, packet_size - len(base_data) - 10)  # Leave room for counter
        padding = "X" * padding_size
        
        print("Starting optimized packet transmission...")
        print("Legend: . = Success, T = Timeout, E = Error, F = Failure")
        print(f"Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        packet_count = 0
        success_count = 0
        last_status_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check duration limit
                if elapsed >= duration:
                    print(f"\nReached duration limit of {duration} seconds.")
                    break
                
                # Create packet with counter
                packet_data = f"{base_data}{packet_count:06d}_{padding}"[:packet_size]
                packet = packet_data.encode('utf-8')
                
                # Send packet
                if self.send_data(packet):
                    success_count += 1
                
                packet_count += 1
                
                # Status update every 5 seconds
                if current_time - last_status_time >= 5.0:
                    success_rate = (success_count / packet_count) * 100 if packet_count > 0 else 0
                    actual_rate = packet_count / elapsed if elapsed > 0 else 0
                    print(f"\n[{elapsed:.1f}s] Packets: {packet_count} | "
                          f"Success: {success_count} ({success_rate:.1f}%) | "
                          f"Rate: {actual_rate:.1f} pps")
                    last_status_time = current_time
                
                # Precise timing control
                loop_time = time.time() - current_time
                sleep_time = max(0, interval - loop_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\nStopped by user after {time.time() - start_time:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Unexpected error during transmission: {e}")
            
        finally:
            # Final statistics
            elapsed = time.time() - start_time
            success_rate = (success_count / packet_count) * 100 if packet_count > 0 else 0
            actual_rate = packet_count / elapsed if elapsed > 0 else 0
            
            print("\n" + "="*60)
            print("Transmission Complete")
            print(f"Duration: {elapsed:.1f} seconds")
            print(f"Total packets: {packet_count}")
            print(f"Successful packets: {success_count}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average rate: {actual_rate:.1f} packets/second")
            print("="*60 + "\n")
            
            # Close socket
            try:
                self.socket.close()
            except:
                pass
            
            # Restore original network connection
            if self.return_to_original:
                print("Restoring original network connection...")
                self.disconnect_and_restore()
            
            return True

def main():
    parser = argparse.ArgumentParser(
        description='Cross-platform Active Station for WiFi CSI collection\n\n'
        'This script will:\n'
        '1. Connect to the target WiFi network\n'
        '2. Optimize settings for small UDP packet transmission\n'
        '3. Send packets to trigger CSI collection at the AP\n'
        '4. Restore the original network connection\n\n'
        'Supported platforms: Linux, macOS, Raspberry Pi\n',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--target-ssid', type=str, required=True,
                      help='SSID of the target WiFi network for CSI collection')
    parser.add_argument('--target-password', type=str, required=True,
                      help='Password for the target WiFi network')
    parser.add_argument('--server-ip', type=str, default='192.168.4.1',
                      help='IP address of the CSI collection AP (default: 192.168.4.1)')
    parser.add_argument('--server-port', type=int, default=80,
                      help='Port of the CSI collection AP (default: 80)')
    parser.add_argument('--channel', type=int, default=1,
                      help='WiFi channel (default: 1)')
    parser.add_argument('--interface', type=str, default=None,
                      help='Network interface (auto-detected if not specified)')
    parser.add_argument('--interval', type=float, default=0.01,
                      help='Time between packets in seconds (default: 0.01 = 100 pps)')
    parser.add_argument('--duration', type=int, default=60,
                      help='Run duration in seconds (default: 60)')
    parser.add_argument('--packet-size', type=int, default=64,
                      help='Size of UDP packets in bytes (default: 64)')
    parser.add_argument('--no-restore', action='store_true',
                      help='Do not restore original network connection')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.interval < 0.001:
        logger.warning("Very small interval may cause system instability")
    
    if args.packet_size < 16 or args.packet_size > 1472:
        logger.warning("Packet size should be between 16-1472 bytes for optimal performance")
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment if not provided
    target_ssid = args.target_ssid or os.getenv('TARGET_WIFI_SSID')
    target_password = args.target_password or os.getenv('TARGET_WIFI_PASSWORD')
    
    if not target_ssid or not target_password:
        logger.error("Target WiFi SSID and password are required")
        sys.exit(1)
    
    print(f"\nTarget Network: {target_ssid}")
    print(f"Packet Rate: {1/args.interval:.1f} packets/second")
    print(f"Duration: {args.duration} seconds")
    print(f"Restore Original: {'No' if args.no_restore else 'Yes'}")
    
    # Create and run the active station
    station = ActiveStation(
        target_ssid=target_ssid,
        target_password=target_password,
        server_ip=args.server_ip,
        server_port=args.server_port,
        channel=args.channel,
        interface=args.interface,
        return_to_original=not args.no_restore
    )
    
    station.debug = args.debug
    
    try:
        success = station.run(
            interval=args.interval,
            duration=args.duration,
            packet_size=args.packet_size
        )
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
