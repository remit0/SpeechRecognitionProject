#!/bin/bash -
#PBS -l nodes=gpu05

echo START
export ROOT_DIR=/vol/gpudata/rar2417
export CUDA_HOME=/vol/gpudata/cuda/8.0.61
export PATH=${PATH}:${CUDA_HOME}/bin
source ${ROOT_DIR}/venv/project/bin/activate

python3 ${SRC_PATH}/combine_models.py -k 1

echo END
