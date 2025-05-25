#!/bin/bash
# CSI Data Collection Script
# Usage: ./csi_data.sh [serial_port] [output_file]
# Example: ./csi_data.sh /dev/ttyUSB0 datalog.csv

set -e

show_help() {
  echo "Usage: $0 [serial_port] [output_file]"
  echo "Example: $0 /dev/ttyUSB0 datalog.csv"
  exit 1
}

# Help option
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_help
fi

# Default values
PORT="/dev/ttyUSB0"
OUTFILE="datalog_$(date +%Y%m%d_%H%M%S).csv"

# Parse arguments
if [ ! -z "$1" ]; then
  PORT="$1"
fi
if [ ! -z "$2" ]; then
  OUTFILE="$2"
fi


# Move to firmware directory
FIRMWARE_DIR="$(dirname "$0")/../firmware/active_sta"
if [ ! -d "$FIRMWARE_DIR" ]; then
  echo "Firmware directory $FIRMWARE_DIR does not exist."
  exit 1
fi
cd "$FIRMWARE_DIR"

# Check if idf.py is available
if ! command -v idf.py >/dev/null 2>&1; then
    echo "idf.py not found in PATH after sourcing ESP-IDF."
    exit 1
fi

# Warn if serial port does not exist
if [ ! -e "$PORT" ]; then
    echo "Warning: Serial port $PORT does not exist. Proceeding anyway."
fi

# Print info
echo "Running: idf.py -p $PORT monitor | tee $OUTFILE"

# Start monitoring and save output
idf.py -p "$PORT" monitor | tee "$OUTFILE"

# End of script
