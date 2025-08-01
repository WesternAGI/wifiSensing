# Cross-Platform Active Station for WiFi CSI Collection

This is a Python implementation of an Active Station (STA) that works on Linux, macOS, and Raspberry Pi. It automatically connects to a target WiFi network, optimizes settings for small UDP packet transmission to maximize CSI (Channel State Information) collection, then restores the original network connection.

## Features

- **Cross-platform support**: Linux, macOS, and Raspberry Pi
- **Automatic OS detection**: Detects Raspberry Pi specifically
- **Network management**: Connects to target network and restores original connection
- **Optimized transmission**: Configures system for high-rate small packet transmission
- **Multiple WiFi backends**: Uses NetworkManager (nmcli), wpa_supplicant, or native macOS tools
- **Flexible configuration**: Command-line arguments and environment variable support

## Prerequisites

- Python 3.6 or higher
- Linux, macOS, or Raspberry Pi OS
- WiFi interface
- Administrator/root privileges (for network optimization)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd wifiSensing/firmware/active_sta_py
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
# Connect to target network and send packets for CSI collection
python active_sta.py --target-ssid TARGET_NETWORK --target-password TARGET_PASSWORD

# Use environment variables (create .env file)
echo "TARGET_WIFI_SSID=your_target_network" > .env
echo "TARGET_WIFI_PASSWORD=your_target_password" >> .env
python active_sta.py --target-ssid TARGET_NETWORK --target-password TARGET_PASSWORD
```

### Advanced Options

```bash
# High-rate transmission (1000 packets/second for 30 seconds)
python active_sta.py \
  --target-ssid TARGET_NETWORK \
  --target-password TARGET_PASSWORD \
  --interval 0.001 \
  --duration 30 \
  --packet-size 64

# Specify server details and interface
python active_sta.py \
  --target-ssid TARGET_NETWORK \
  --target-password TARGET_PASSWORD \
  --server-ip 192.168.1.100 \
  --server-port 8080 \
  --interface wlan0

# Don't restore original network connection
python active_sta.py \
  --target-ssid TARGET_NETWORK \
  --target-password TARGET_PASSWORD \
  --no-restore
```

### Command-Line Arguments

- `--target-ssid`: SSID of target WiFi network (required)
- `--target-password`: Password for target network (required)
- `--server-ip`: IP address of CSI collection AP (default: 192.168.4.1)
- `--server-port`: Port of CSI collection AP (default: 80)
- `--interval`: Time between packets in seconds (default: 0.01 = 100 pps)
- `--duration`: Run duration in seconds (default: 60)
- `--packet-size`: UDP packet size in bytes (default: 64)
- `--interface`: Network interface (auto-detected if not specified)
- `--no-restore`: Don't restore original network connection
- `--debug`: Enable debug logging

## How It Works

1. **OS Detection**: Automatically detects Linux, macOS, or Raspberry Pi
2. **Network Backup**: Saves current network connection for later restoration
3. **Target Connection**: Connects to the specified target WiFi network
4. **Optimization**: Configures system settings for optimal small packet transmission
5. **Packet Transmission**: Sends UDP packets at specified rate to trigger CSI collection
6. **Network Restoration**: Reconnects to the original network (unless --no-restore is used)

## Platform-Specific Notes

### Linux/Raspberry Pi
- Uses NetworkManager (nmcli) as primary method
- Falls back to wpa_supplicant if NetworkManager unavailable
- Requires sudo privileges for network optimization

### macOS
- Uses native networksetup commands
- Requires admin privileges for network changes
- May prompt for password during network switching

## Troubleshooting

### Connection Issues
- Ensure target network credentials are correct
- Check that WiFi interface is available and enabled
- Verify no conflicting network managers are running

### Permission Issues
```bash
# On Linux/Raspberry Pi, run with sudo for full optimization
sudo python active_sta.py --target-ssid NETWORK --target-password PASSWORD

# On macOS, you may be prompted for admin password
```

### Network Restoration Issues
- Original network may require manual reconnection if password changed
- Use --no-restore flag if you want to stay on target network
- Check system network preferences if automatic restoration fails

## Example Output

```
Target Network: MyTargetWiFi
Packet Rate: 100.0 packets/second
Duration: 60 seconds
Restore Original: Yes

============================================================
Active Station for WiFi CSI Collection
OS: Linux
Interface: wlan0
Target Network: MyTargetWiFi
Target AP: 192.168.4.1:80
Packet Rate: 100.0 packets/second
Packet Size: 64 bytes
Duration: 60 seconds
============================================================

Starting optimized packet transmission...
Legend: . = Success, T = Timeout, E = Error, F = Failure
Press Ctrl+C to stop early

....................

[5.0s] Packets: 500 | Success: 485 (97.0%) | Rate: 100.0 pps
....................

============================================================
Transmission Complete
Duration: 60.0 seconds
Total packets: 6000
Successful packets: 5820
Success rate: 97.0%
Average rate: 100.0 packets/second
============================================================

Restoring original network connection...
```

## Environment Variables

Create a `.env` file for convenience:

```bash
TARGET_WIFI_SSID=your_target_network
TARGET_WIFI_PASSWORD=your_target_password
```

## Performance Tips

- Use `--interval 0.001` for maximum packet rate (1000 pps)
- Smaller packet sizes (32-64 bytes) work best for CSI collection
- Run with sudo/admin privileges for optimal network performance
- Monitor system resources during high-rate transmission
