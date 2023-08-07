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
      alsa-utils


RUN apt-get install -y --no-install-recommends \
      gcc cython3 make

RUN mkdir /src
WORKDIR /src

# COPY requirements.txt requirements.txt

COPY . ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN make clean && make

ENV QT_X11_NO_MITSHM=1

# RUN gcc -o run main.c

# CMD ["./run"]

