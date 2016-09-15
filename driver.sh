#!/bin/bash

set -e

. setup_gpu.sh

python main.py -f eeg -a elu || echo 1 failed
python main.py -f eeg -a relu || echo 2 failed 
python main.py -f piezo -a elu || echo 3 failed
python main.py -f piezo -a relu || echo 4 failed

python evaluate.py -f eeg -a elu || echo 5 failed
python evaluate.py -f eeg -a relu || echo 6 failed
python evaluate.py -f piezo -a elu || echo 7 failed
python evaluate.py -f piezo -a relu || echo 8 failed

