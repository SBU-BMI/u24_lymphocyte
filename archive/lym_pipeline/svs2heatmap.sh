#!/bin/bash
# This script will execute for days, run it with nohup!

source ./conf/bashrc_base.sh

# Break images under svs/ to png tiles
cd patch_extraction
bash start.sh
cd ..

# Apply the trained CNN to predict all patches
cd prediction
bash start.sh
cd ..

# Generate heatmap json files
cd heatmap_gen
bash start.sh
cd ..

exit 0
