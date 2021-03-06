#!/bin/bash -
#PBS -l nodes=gpu05

echo START
export ROOT_DIR=/vol/gpudata/rar2417
export CUDA_HOME=/vol/gpudata/cuda/8.0.61
export PATH=${PATH}:${CUDA_HOME}/bin
export SRC_PATH=${ROOT_DIR}/src/model4
export DATA_PATH=${ROOT_DIR}/Data
export OUTPUT_PATH=${ROOT_DIR}/results/model4
source ${ROOT_DIR}/venv/project/bin/activate

python3 ${SRC_PATH}/submission_cnn.py

echo END
