#!/bin/bash

nvidia-docker run \
    --name lym-pipeline-container -it \
    -v `pwd`/log/:/home/lym_pipeline/log/ \
    -v `pwd`/conf/:/home/lym_pipeline/conf/ \
    -v `pwd`/data/:/home/lym_pipeline/data/ \
    -v `pwd`/svs/:/home/lym_pipeline/svs/ \
    -v `pwd`/patches/:/home/lym_pipeline/patches/ \
    -v `pwd`/models_training/:/home/lym_pipeline/training/models/ \
    -v `pwd`/models_prediction/:/home/lym_pipeline/prediction/models/ \
    -v `pwd`/heatmap_jsons/:/home/lym_pipeline/heatmap_gen/json/ \
    -d lehou0312/lym-pipeline-image-v0

exit 0
