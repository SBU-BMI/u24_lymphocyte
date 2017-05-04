#!/bin/bash

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
