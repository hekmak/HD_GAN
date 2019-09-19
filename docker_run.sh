#!/usr/bin/env bash
# Maintained by H.H

nvidia-docker run --name=tensorflow14 \
    -p 8885:8888 \
    -p 6001:6006 \
    -v /home/hamid/Projects/GANs/ham_gan:/notebooks/project \
    -v /media/hamid/hamid_drive1/Datasets:/notebooks/dataset \
    -it \
    -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    hamidhekmatian/tensorflow:1.14.0 \
    bash
exit

