# WifiSensing



## Quick Start 

1- Clone this repository 
2- Install the espressif tool chain for Linux/Mac ( v4.3)
3- Connect the two esp32 MCUs and run  ls /dev/ttyUSB* to check where they are connected. 
4- Open two terminals (one for each MCU) and run get_idf in each to activate esp-idf in that terminal. 
5- In each project directory, run idf.py build
6- For the active_ap, run idf.py -p /dev/ttyUSB0 flash monitor | grep "CSI_DATA" >> ../../data/<filename>.csv
7- For the active_sta, run idf.py -p /dev/ttyUSB1 flash 
