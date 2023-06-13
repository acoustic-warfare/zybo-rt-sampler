#!/bin/sh
echo "Creating script for uBoot"
mkimage -A arm -O linux -T script -C none -a 0 -e 0 -n 'Execute uImage.bin' -d boot/boot.cmd boot.scr

echo "Copy boot.scr to your boot/"
