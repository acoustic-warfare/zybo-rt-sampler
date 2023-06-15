# Information (Under development)

Rapid Development on the Zybo using Baremetal uBoot or Arch Linux

You may copy the contents of `out/` to the SD-card with `out/boot/` on the boot partition and so on.


## Prerequisites (Ubuntu)

### 1. A DHCP Server

Install the DHCP server:

```
sudo apt install isc-dhcp-server
```

Determine correct interface by looking at `ip a` and check for you interface e.g. `enh0`

Configure `/etc/default/isc-dhcp-server` by setting `INTERFACESv4=<YOUR_INTERFACE>`

Modify `/etc/dhcp/dhcpd.conf` to:

```
# a simple /etc/dhcp/dhcpd.conf
default-lease-time 600;
max-lease-time 7200;
authoritative;

subnet 10.0.0.0 netmask 255.255.255.0 {
 range 10.0.0.2 10.0.0.20;
}
```

Set your server IP address:

```
sudo ifconfig <YOUR_INTERFACE> 10.0.0.1
```

Enable and restart the server:

```
sudo systemctl enable isc-dhcp-server
sudo systemctl restart isc-dhcp-server
```




### 2. A TFTP Server

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

### 3. Configured SD-card

Run the script

```
chmod +x configure.sh
./configure.sh
```


# TODO

This is an installation for setting up data acquisition on the Zybo z7 development board using U-Boot and Linux

## Requirements

Follow the steps in https://github.com/f4pga/symbiflow-xc7z-automatic-tester to create a bootable image for the Zybo z7

## Building

TODO

## Steps

## Connecting to the Zybo via SSH
1. Wait for the process "crng init" to finish. Usually takes between 5 to 15 minutes.
2. Enable SSHD:
```bash
systemctl enable sshd
systemctl enable sshd.service
```
3. Connect to the Zybo with 
```bash
ssh root@ip_address
```

## Known Issues

### "A start job is running for Network Name Resolution"
A bug was found during boot of Arch linux on the Zybo: "A start job is running for Network Name Resolution" which prevented the system from booting. A solution was found by removing the service:

1. In the /root/ folder, disable /etc/systemd/system/sysinit.target.wants/systemd-resolved.service by removing the symlink

## TODO

Set ip address in arch:

```bash
ip address add 10.0.0.2/24 dev end0
```

```bash
arm-none-eabi-gcc -lgcc -lc -lm demo.c --specs=nosys.specs -o demo.elf
arm-none-eabi-objcopy -O binary demo.out demo.bin
cp demo.bin /data/tftp/
```
