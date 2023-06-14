#!/bin/sh

# Startup script after boot to initiate user program. Requires a working ethernet connection
cd /home/alarm
tftp 10.0.0.1 69 -c get program /home/alarm/program
./program