# Information

This repository contains programs, files and tips on how to develop and use the transmitted data from the Zybo Z7-20. It is the second stage in the data pipeline for the Zybo and over to a PC.

The following directories can be summarized as the following:

* `zybo/` Scripts for creating UBoot images and setup a Docker TFTP server on the PC.

* `out/` Generated images from the scripts in zybo/

* `PC/` A large collection of APIs, programs and tips on how to use the microphone array. It is here the beamforming algorithms resides.

Rapid Development on the Zybo using Baremetal uBoot


# Prerequisites

## TFTP Server
In order for the Zybo Z7 to boot using tftpboot, a running TFTP server is required. It can be configured, but the Zybo is initially looking for two files: `bitstream` and `ps.elf` generated by Vivado/Xilinx SDK. The Zybo will look at `10.0.0.1` and the default TFTP port `69`, but this can be configured in `zybo/`.

### Docker (Recommended)
It is not required to use a docker container, however a `Dockerfile` has been configured for a TFTP server located in `zybo/Dockerfile`. Please read the `zybo/README.md` before continuing.

Move to the Dockerfile:
```
cd zybo
```

Build the Docker-image:
```
dockerbuild -t tftpd .
```

Start the container:
```
docker run -it -d --rm --network=host -v YOUR_PATH:/var/lib/tftpboot --name tftp-server tftpd
```

Stop the service
```
docker kill tftp-server
```

## Setup (Ubuntu)

### 1. A TFTP Server
See: TFTP

### TFTPD-HPA
Install the TFTP server:

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