#!/bin/bash

pushd ../

set -e

NGPU=2

. setup_gpu.sh

rm -rf checkpoint/*/output.csv

# run multiple jobs in parallel, but don't run more than one job on a
# single GPU
parallel --verbose -P ${NGPU} --line-buffer --rpl '{gpuid} $_ = $job->slot() - 1' \
  CUDA_VISIBLE_DEVICES={gpuid} \
  ./driver/train_test.sh {} :::: ./driver/methods.list

