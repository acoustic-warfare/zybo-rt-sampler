# docker build -t aw .

# 

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y

# ENV LANG en_US.utf8

RUN apt-get install gcc -y

RUN apt-get install -y --no-install-recommends \
      xcb \
      python3-dev \
      portaudio19-dev \
      python3-pip \
      libglib2.0-0 \
      libgl1-mesa-glx \
      libasound2-dev \
      build-essential \
      libeigen3-dev \
      alsa-utils

# RUN cd ~/usr/include && \
#       ln -sf eigen3/Eigen Eigen && \
#       ln -sf eigen3/unsupported unsupported


RUN apt-get install -y --no-install-recommends \
      gcc cython3 make

RUN mkdir /src
WORKDIR /src

COPY . ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt # --no-cache-dir 

# RUN make clean && make

ENV QT_X11_NO_MITSHM=1

