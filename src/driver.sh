#!/bin/bash

set -e

. setup.sh

python evaluate.py -n v2 -a relu -c 1
python evaluate.py -n v2.1 -a relu -c 1
python evaluate.py -n v2.2 -a relu -c 1
python evaluate.py -n v3 -a relu -c 1

