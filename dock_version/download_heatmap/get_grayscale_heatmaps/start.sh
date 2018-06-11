#!/bin/bash

source ../../conf/variables.sh

bash get_grayscale_heatmap.sh &> ${LOG_OUTPUT_FOLDER}/log.get_grayscale_heatmaps.txt

exit 0
