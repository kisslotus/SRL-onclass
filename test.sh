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
if [ $# == 0 ] ; then
python test.py
elif [ $# == 2 ] ; then
python test.py $1 $2
elif [ $# == 4 ] ; then
python test.py $1 $2 $3 $4
else
echo "Invalid params！"
fi

