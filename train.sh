#!/bin/sh

if [ ! -d "model_info" ]; then
    mkdir model_info
fi
if [ ! -d "model_info/struct" ]; then
    mkdir model_info/struct
fi
if [ ! -d "model_info/weights" ]; then
    mkdir model_info/weights
fi
if [ ! -d "output" ]; then
    mkdir output
fi
if [ ! -d "output/logs" ]; then
    mkdir output/logs
fi

python preprocess.py
python train.py
