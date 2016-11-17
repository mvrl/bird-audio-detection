ANACONDA_HOME=/u/amo-d0/csfac/jacobs/anaconda
LD_LIBRARY_PATH+=:/u/amo-d0/lib/cuda-8.0_cudnn-v5.1/lib64
LD_LIBRARY_PATH+=:$ANACONDA_HOME/lib

export LD_LIBRARY_PATH

source activate donohue_gpu

