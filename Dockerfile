FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-rosbridge-server \
    ros-humble-web-video-server \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    mediapipe \
    pyzbar \
    numpy \
    pandas

WORKDIR /ws

CMD ["bash"]