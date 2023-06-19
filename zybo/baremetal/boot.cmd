setenv serverip 10.0.0.1
setenv ipaddr 10.0.0.2

tftpboot 0x4000000 bitstream
fpga loadb 0 0x4000000 $filesize

tftpboot 0x0 ps.elf
bootelf -p 0x0