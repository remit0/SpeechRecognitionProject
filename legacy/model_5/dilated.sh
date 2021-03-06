#!/bin/bash -
#PBS -l nodes=gpu16

echo START
export ROOT_DIR=/vol/gpudata/rar2417
export CUDA_HOME=/vol/gpudata/cuda/8.0.61
export PATH=${PATH}:${CUDA_HOME}/bin
export SRC_PATH=${ROOT_DIR}/src/model5
export DATA_PATH=${ROOT_DIR}/Data
export OUTPUT_PATH=${ROOT_DIR}/results/model5
source ${ROOT_DIR}/venv/project/bin/activate

python3 ${SRC_PATH}/dilated.py -lr 0.0006 -ld 0.87 -wt 1 -md 4 -key 13

echo END