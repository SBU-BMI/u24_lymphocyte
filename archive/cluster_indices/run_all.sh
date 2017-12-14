#!/bin/bash

cat $1 | /home/tkurc/programs/gnu-parallel/bin/parallel -j $2 ./run_clust.sh 
