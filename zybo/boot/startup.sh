#!/bin/sh

# Startup script after boot to initiate user program. Requires a working ethernet connection
cd /home/alarm
tftp 10.0.0.1 69 -c get program /home/alarm/program
#Uncomment line bellow if permission denied
#chmod a=rwx /home/alarm/program
./program