ANACONDA_HOME=/u/amo-d0/csfac/jacobs/anaconda
LD_LIBRARY_PATH+=:~/lib/cuda-8.0_cudnn-v5.1/lib64
LD_LIBRARY_PATH+=:$ANACONDA_HOME/lib

export LD_LIBRARY_PATH

TF_CPP_MIN_LOG_LEVEL=2
export TF_CPP_MIN_LOG_LEVEL

source activate tensorflow_gpu

