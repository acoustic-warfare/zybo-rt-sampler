
## Requirements (Ubuntu 22.04)
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
```