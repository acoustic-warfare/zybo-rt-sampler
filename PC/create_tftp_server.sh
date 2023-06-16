#!/bin/sh

# A script for setting up a TFTP server for the Zybo to get its binaries from

echo "\n\n"\
"Creating directory for putting your 'ps.elf' and 'bitstream' files at: \n\n/data/tftp/\n"

echo "Installing TFTP binaries"

sudo apt install tftpd-hpa

echo "Generating TFTP directory"
sudo mkdir /data
sudo mkdir /data/tftp

sudo chmod -R 777 /data/tftp
sudo chown -R nobody /data/tftp


sudo cat >/etc/default/tftpd-hpa <<EOL
# /etc/default/tftpd-hpa

TFTP_USERNAME="tftp"
TFTP_DIRECTORY="/data/tftp"
TFTP_ADDRESS=":69"
TFTP_OPTIONS="--secure -v"
EOL

sudo systemctl enable tftpd-hpa
sudo systemctl restart tftpd-hpa

echo "Done"
