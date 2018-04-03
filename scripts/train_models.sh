#!/bin/bash

cd ../
cd training/autoencoder
bash start_cae_training.sh
cd ../..

cd training/lymphocyte
bash start_cnn_lymphocyte_training.sh
cd ../..

cd training/necrosis
bash start_cnn_necrosis_training.sh
cd ../..

exit 0
