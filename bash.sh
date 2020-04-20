#!/bin/bash

if [ -d '/proc/driver/nvidia' ]; then
    docker run --rm -u $(id -u):$(id -g) -v $(pwd):/home/translator -it --gpus all griffith/mwp-automatic-solver bash
    else
    docker run --rm -u $(id -u):$(id -g) -v $(pwd):/home/translator -it griffith/mwp-automatic-solver bash
fi