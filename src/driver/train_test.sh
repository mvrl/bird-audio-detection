#!/bin/bash

stdbuf -o0 python train.py $1 
stdbuf -o0 python evaluate.py $1 

