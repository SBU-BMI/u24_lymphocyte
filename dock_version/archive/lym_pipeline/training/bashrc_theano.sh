module load cuda75
export PATH=/home/lehhou/git/bin/:${PATH}
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cm/shared/apps/anaconda2/current/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cm/shared/apps/cuda75/toolkit/7.5.18/lib64/"
export CUDA_HOME=/cm/shared/apps/cuda75
source ~/theano/bin/activate

