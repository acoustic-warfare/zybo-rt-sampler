# Information

This is an installation for setting up data acquisition on the Zybo z7 development board using U-Boot and Linux

## Requirements

Follow the steps in https://github.com/f4pga/symbiflow-xc7z-automatic-tester to create a bootable image for the Zybo z7

## Building

TODO

## Steps

## Known Issues

### "A start job is running for Network Name Resolution"
A bug was found during boot of Arch linux on the Zybo: "A start job is running for Network Name Resolution" which prevented the system from booting. A solution was found by removing the service:

1. In the /root/ folder, disable /etc/systemd/system/sysinit.target.wants/systemd-resolved.service by removing the symlink

## TODO

```bash
arm-none-eabi-gcc -lgcc -lc -lm demo.c --specs=nosys.specs -o demo.elf
arm-none-eabi-objcopy -O binary demo.out demo.bin
cp demo.bin /data/tftp/
```
