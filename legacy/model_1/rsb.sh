#!/bin/bash -
#PBS -l nodes=gpu12

echo START
export ROOT_DIR=/vol/gpudata/rar2417
export CUDA_HOME=/vol/gpudata/cuda/8.0.61
export PATH=${PATH}:${CUDA_HOME}/bin
export SRC_PATH=${ROOT_DIR}/src/model1
export DATA_PATH=${ROOT_DIR}/Data
export OUTPUT_PATH=${ROOT_DIR}/results/model1
source ${ROOT_DIR}/venv/project/bin/activate

python3 ${SRC_PATH}/rsb.py -lr 0.003 -ld 0.87 -wt 1 -ft 512 -nl 2 -md 1 -key 1

echo END