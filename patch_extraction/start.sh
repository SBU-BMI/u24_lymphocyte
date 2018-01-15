#!/bin/bash

source ../conf/variables.sh

nohup bash save_svs_to_tiles.sh 0 4 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_0.txt &
nohup bash save_svs_to_tiles.sh 1 4 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_1.txt &
nohup bash save_svs_to_tiles.sh 2 4 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_2.txt &
nohup bash save_svs_to_tiles.sh 3 4 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_3.txt &
wait

exit 0
