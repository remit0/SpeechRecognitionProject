#!/bin/bash -
#PBS -l nodes=gpu01

echo START
export ROOT_DIR=/vol/gpudata/rar2417
export CUDA_HOME=/vol/gpudata/cuda/9.1.85
export PATH=${PATH}:${CUDA_HOME}/bin
export SRC_PATH=${ROOT_DIR}/src/model3
export DATA_PATH=${ROOT_DIR}/Data
export OUTPUT_PATH=${ROOT_DIR}/results/model3

source ${ROOT_DIR}/venv/project/bin/activate
python3 ${SRC_PATH}/spec.py -lr 0.0001 -ld 0.87 -wt 1 -ft 512 -nl 2 -key 1
echo END
