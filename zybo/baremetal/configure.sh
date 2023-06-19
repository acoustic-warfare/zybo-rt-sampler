#!/bin/sh

if ! [ -x "$(command -v mkimage)" ]; then
  echo 'Error: Please install mkimage' >&2
  exit 1
fi


echo "Generating baremetal bootable medium..."

echo "Generating uBoot boot-script..."

mkimage -A arm -O linux -T script -C none -a 0 -e 0 -n 'uImage boot-script' -d boot.cmd ../../out/boot/boot.scr

echo "Created: boot.scr"