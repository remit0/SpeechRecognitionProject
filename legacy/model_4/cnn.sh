#!/bin/bash -
#PBS -l nodes=gpu13

echo START
export ROOT_DIR=/vol/gpudata/rar2417
export CUDA_HOME=/vol/gpudata/cuda/9.1.85
export PATH=${PATH}:${CUDA_HOME}/bin
export SRC_PATH=${ROOT_DIR}/src/model4
export DATA_PATH=${ROOT_DIR}/Data
export OUTPUT_PATH=${ROOT_DIR}/results/model4

source ${ROOT_DIR}/venv/project/bin/activate
python3 ${SRC_PATH}/cnn.py -lr 0.001 -ld 0.87 -key 6
echo END
