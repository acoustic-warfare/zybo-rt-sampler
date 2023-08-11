#!/bin/bash 

echo "Starting Docker"

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

docker run -m 8GB -it --rm --network=host \
    --device=/dev/snd:/dev/snd --device=/dev/video0 -v /usr/share/alsa:/usr/share/alsa \
    -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v ${PWD}:/src ${1:-bash}
xhost -local:docker