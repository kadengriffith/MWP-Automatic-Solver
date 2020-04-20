#!/bin/bash

if [ -d '/proc/driver/nvidia' ]; then
    # For selective GPUs use: --gpus '"device=0,1,2,3"'
    docker run --rm -u $(id -u):$(id -g) -v $(pwd):/home/translator -t --gpus all griffith/mwp-automatic-solver ./trial.sh
    else
    docker run --rm -u $(id -u):$(id -g) -v $(pwd):/home/translator -t griffith/mwp-automatic-solver ./trial.sh
fi