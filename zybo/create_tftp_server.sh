#!/bin/sh
echo "Installing TFTP binaries"

sudo apt install tftpd-hpa

echo "Generating TFTP directory"
sudo mkdir /data
sudo mkdir /data/tftp

sudo chmod -R 777 /data/tftp
sudo chown -R nobody /data/tftp


sudo cat >/etc/default/tftpd-hpa2 <<EOL
# /etc/default/tftpd-hpa2

TFTP_USERNAME="tftp"
TFTP_DIRECTORY="/data/tftp"
TFTP_ADDRESS=":69"
TFTP_OPTIONS="--secure -v"
EOL

sudo service tftpd-hpa restart

echo
echo "Add your compiled bitstream named as "bitstream" to :"
echo
echo "/data/tftp/"
echo
echo "Example: /data/tftp/bitstream"
