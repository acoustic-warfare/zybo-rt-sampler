setenv bootargs "root=/dev/mmcblk0p2 rw rootwait"
setenv bootcmd "load mmc 0 0x1000000 uImage && load mmc 0 0x2000000 devicetree.dtb && bootm 0x1000000 - 0x2000000"
boot