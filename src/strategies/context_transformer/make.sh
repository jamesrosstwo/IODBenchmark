#!/usr/bin/env bash
cd ./utils/

which python

CUDA_PATH=/usr/local/cuda/

python build.py build_ext --inplace --force

cd ..
