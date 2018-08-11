echo START
export ROOT_DIR=/home/r2d9/Desktop/SpeechRecognitionProject/refactored
export SRC_PATH=${ROOT_DIR}

python3 ${SRC_PATH}/training.py -lr 0.0003 -key 1

echo END