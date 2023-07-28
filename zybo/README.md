# Information

These are files gathered for the FPGA to boot from a remote .elf and bitstream

The docker container creates an tftp server on the port 69 as described in `tftpd-hpa`

## Setup (Ubuntu)

### 1. A TFTP Server

Install the FTP server:

```
sudo apt install tftpd-hpa
```

Create the directory where the board will look for its files:

```
sudo mkdir -p /data/tftp
sudo chmod -R 777 /data/tftp
sudo chown -R nobody /data/tftp
```

Configure TFTP to your specified directory by adding the following content to `/etc/default/tftpd-hpa`:
```
# /etc/default/tftpd-hpa

TFTP_USERNAME="tftp"
TFTP_DIRECTORY="/data/tftp"
TFTP_ADDRESS=":69"
TFTP_OPTIONS="--secure -v"
```

Enable and restart the server:

```
sudo systemctl enable tftpd-hpa
sudo systemctl restart tftpd-hpa
```

### 2. Configured SD-card
1. Follow the steps in https://github.com/f4pga/symbiflow-xc7z-automatic-tester to configure the SD card with U-Boot

2. Run the script
```
./zybo/baremetal/configure.sh
```
3. Copy ./out/boot.scr to boot partition of SD-card

### 3. Bitstream and ELF
Copy your bitstream and ps.elf file to /data/tftp on your computer

### 4. Set a static IP-address
Set the ip address of the ethernet port that the zybo is connected to:
```
IPv4 address: 10.0.0.1
Netmask: 255.255.255.0
Gateway: 192.168.1.1
```

### 5. Requirements
python3
cython3
gcc
make

### 6. Config
Edit the config.json file in ./PC/src to your desired values

### 7. Make
Run:
```
make clean
make
```

### 8. Connections
Connect the zybo to the USB and ethernet port

### 9. Run your desired program
Run your desired program from ./PC/interface