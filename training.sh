#!/bin/bash -
#PBS -l nodes=gpu12

echo START
export ROOT_DIR=/vol/gpudata/rar2417
#export CUDA_HOME=/vol/gpudata/cuda/9.1.85
. /vol/gpudata/cuda/9.1.85/setup.sh
export PATH=${PATH}:${CUDA_HOME}/bin
export SRC_PATH=${ROOT_DIR}/src
export DATA_PATH=${ROOT_DIR}/Data
export OUTPUT_PATH=${ROOT_DIR}/results

source ${ROOT_DIR}/venv/project/bin/activate
#python ${SRC_PATH}/training_doc.py -source_path ${SRC_PATH} -data_path ${DATA_PATH} -output_path ${OUTPUT_PATH} -model ${OUTPUT_PATH}/models/model_save_ResNet_1.ckpt -e 20 -b 20 -lr 0.003 -ft 512 -nl 2 -mode 1
python ${SRC_PATH}/training_doc_v2.py -mode 1
echo END