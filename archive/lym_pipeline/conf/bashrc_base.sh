# .bashrc

module load anaconda/2.1.0
module load cuda55/toolkit cluster-tools 
module load sge 
module load mvapich2/gcc/64/1.9-mlnx-ofed2
module load matlab/mcr-2014b
module load cmake30
module load gcc
module load mongodb/3.2.0
module load jdk8/1.8.0_11
module load openslide/3.4.0
module load extlibs/1.0.0
module load ITK/4.6.1
module load caffe/caffe
export LIBTIFF_CFLAGS="-I/cm/shared/apps/extlibs/include" 
export LIBTIFF_LIBS="-L/cm/shared/apps/extlibs/lib -ltiff" 
module load mongodb/3.2.0
module load anaconda2/4.4.0

