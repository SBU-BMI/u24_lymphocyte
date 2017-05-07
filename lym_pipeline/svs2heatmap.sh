#!/bin/bash

cd camic_setup
bash copy_svs_to_camic.sh
bash put_meta_data_in.sh
bash touch_humanmark.sh
cd ..

cd patch_extraction
bash start.sh
cd ..

cd prediction
bash start.sh
cd ..

cd heatmap_gen
bash start.sh
cd ..

exit 0
