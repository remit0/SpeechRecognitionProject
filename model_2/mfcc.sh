#!/bin/bash -
#PBS -l nodes=gpu12

echo START
export ROOT_DIR=/vol/gpudata/rar2417
export CUDA_HOME=/vol/gpudata/cuda/9.1.85
export PATH=${PATH}:${CUDA_HOME}/bin
export SRC_PATH=${ROOT_DIR}/src/model2
export DATA_PATH=${ROOT_DIR}/Data
export OUTPUT_PATH=${ROOT_DIR}/results/model2

source ${ROOT_DIR}/venv/project/bin/activate
python ${SRC_PATH}/mfcc.py -key 1
echo END
# 1 : lr 0.003 features 512 layers 2 step_size 5