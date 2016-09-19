#!/bin/bash

set -e

. setup_gpu.sh

python main.py -f eeg -a elu -c 1
python main.py -f eeg -a relu -c 1 
python main.py -f eeg -a lrelu -c 1 
python main.py -f piezo -a elu -c 1 
python main.py -f piezo -a relu -c 1 
python main.py -f piezo -a lrelu -c 1 

python evaluate.py -f eeg -a elu -c 1
python evaluate.py -f eeg -a relu -c 1 
python evaluate.py -f eeg -a lrelu -c 1 
python evaluate.py -f piezo -a elu -c 1 
python evaluate.py -f piezo -a relu -c 1 
python evaluate.py -f piezo -a lrelu -c 1 

python main.py -f eeg -a elu -c 1.2
python main.py -f eeg -a relu -c 1.2
python main.py -f eeg -a lrelu -c 1.2
python main.py -f piezo -a elu -c 1.2
python main.py -f piezo -a relu -c 1.2
python main.py -f piezo -a lrelu -c 1.2

python evaluate.py -f eeg -a elu -c 1.2
python evaluate.py -f eeg -a relu -c 1.2
python evaluate.py -f eeg -a lrelu -c 1.2
python evaluate.py -f piezo -a elu -c 1.2
python evaluate.py -f piezo -a relu -c 1.2
python evaluate.py -f piezo -a lrelu -c 1.2

python main.py -f eeg -a elu -c .8
python main.py -f eeg -a relu -c .8 
python main.py -f eeg -a lrelu -c .8 
python main.py -f piezo -a elu -c .8 
python main.py -f piezo -a relu -c .8 
python main.py -f piezo -a lrelu -c .8 

python evaluate.py -f eeg -a elu -c .8
python evaluate.py -f eeg -a relu -c .8 
python evaluate.py -f eeg -a lrelu -c .8 
python evaluate.py -f piezo -a elu -c .8 
python evaluate.py -f piezo -a relu -c .8 
python evaluate.py -f piezo -a lrelu -c .8 

