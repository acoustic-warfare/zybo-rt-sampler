
## Prerequisites (Ubuntu 22.04)
The program only can be built using libraries located on the system, but
an Docker image has been made for that purpose.

### Docker

The only requirements for Ubuntu is a an available Docker runtime

Follow the required steps in https://docs.docker.com/engine/install/ubuntu/
before proceeding.


### Git (Optional)

Install git:

```bash
sudo apt-get install git
```

## Building the program
This repository contains two parts, the Zybo UBoot image creation and the beamforming application

For building the Zybo parts see `zybo/README.md`

## TFTP-Server (Docker)

To work out of the box, make sure the computer has a static IP of `10.0.0.1` but this can be changed in the configuration, however that address is hardcoded in the ps program located on the Zybo see: https://github.com/acoustic-warfare/FPGA-sampling.

Change directory

```bash
cd zybo
```

Build the docker image:

```bash
docker build -t tftpd .
```

Start the container and change `YOUR_PATH` to where you want to store your bitstreams:

```bash
docker run -it -d --rm --network=host -v YOUR_PATH:/var/lib/tftpboot --name tftp-server tftpd
```

## Beamforming Application (Docker)
Change directory

```bash
cd PC2
```

Build the docker image (aw is just a name, can by anything):
```bash
docker build -t aw .
```



make the starting script executable:
```bash
chmod +x start.sh
```

Start the container. This will mount the current directory and let the user perform building inside the docker image whilst still being able to change configuration from the outside:

```bash
./start.sh aw 
```

If it looks similar to this, then its successful:

```bash
Starting Docker
root@:/src#
```

### Building the program (Inside Docker)

Build the entire application
```bash
make
```

* `listen` C program to listen to the microphones in real-time
* `config` Build the configuration files before compiling the shared objects
* `clean` Remove all build artifacts for a clean build.

### Run the program

A quick demo with both audio-playback/steering and heatmaps can be ran by:

```bash
python3 demo.py
```





<!-- # Build
# dockerbuild -t tftpd .

# Start the container
# docker run -it -d --rm --network=host -v YOUR_PATH:/var/lib/tftpboot --name tftp-server tftpd

 



The requirements can be installed using the following command:

```bash
sudo apt update
sudo apt upgrade -y
```

Building requirements:
```bash
sudo apt install \
            cython3 python3-cython gcc make
```

Libraries:
```bash

sudo apt install \
    python3-numpy python3-opencv python3-matplotlib qtbase5-dev
```

## Building



## Known Issues

* ### shmget not working: Invalid argument
Try editing the flag `KEY` located in `src/config.json`. Remember that the key must at least be +/-2 from where it is now since two shared memory keys are used, one after the other. The error likely stems from wanting to use a shared memory location which has not been cleaned up correctly, likely due to bad stop of the program.

* ### Resource busy
This program spawns multiple processes, and in the event of a crash, some processes may still use up devices (looking at you python). When this happens, issue the command:

```bash
killall python3
``` -->