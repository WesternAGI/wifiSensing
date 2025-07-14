To increase transmission speed, run active_sta.py with a smaller --interval value (e.g., --interval 0.01 for 100 packets/sec):

    python active_sta.py --ssid mywifi_ssid --server-ip 192.168.4.1 --interval 0.01

The script now uses a shorter socket timeout for faster retries.
# Active Station (Python Implementation)

This is a Python implementation of the Active Station (STA) that runs on Linux/macOS. It connects to a WiFi network and sends data to trigger CSI (Channel State Information) collection on an access point.

## Prerequisites

- Python 3.6 or higher
- Linux or macOS
- WiFi interface
- Root/Administrator privileges (for some operations)

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
# Run with SSID and password as command-line arguments
python active_sta.py --ssid YOUR_WIFI_SSID --password YOUR_WIFI_PASSWORD

# Or use environment variables
# On Linux/macOS:
export WIFI_SSID=your_ssid
export WIFI_PASSWORD=your_password
python active_sta.py
```

### Advanced Options

```bash
# Specify server IP and port (default: 192.168.4.1:80)
python active_sta.py --ssid YOUR_SSID --password YOUR_PASSWORD --server-ip 192.168.4.1 --server-port 80

# Set transmission interval in seconds (default: 1.0)
python active_sta.py --ssid YOUR_SSID --password YOUR_PASSWORD --interval 0.5

# Run for a specific duration in seconds
python active_sta.py --ssid YOUR_SSID --password YOUR_PASSWORD --duration 300  # Run for 5 minutes

# Enable debug logging
python active_sta.py --ssid YOUR_SSID --password YOUR_PASSWORD --debug
```

### Using a .env File

Create a `.env` file in the same directory with the following content:

```
WIFI_SSID=your_wifi_ssid
WIFI_PASSWORD=your_wifi_password
```

Then run:

```bash
python active_sta.py
```

## How It Works

1. The script connects to the specified WiFi network.
2. It sends UDP packets to the specified server (AP) at regular intervals.
3. The AP can use these packets to collect CSI data.

## Logging

Logs are written to `active_sta.log` in the same directory.

## Troubleshooting

- **Connection Issues**: Ensure the WiFi credentials are correct and the network is in range.
- **Permission Denied**: Run the script with `sudo` on Linux if you encounter permission errors.
- **Interface Not Found**: The script tries to auto-detect the WiFi interface. If it fails, you may need to modify the `_get_wifi_interface` method in the script.

## License

This project is part of the WiFi Sensing project.
