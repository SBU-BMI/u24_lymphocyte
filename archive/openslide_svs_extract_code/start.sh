#!/bin/bash

nohup bash save_svs_to_tiles.sh 0 4 &
nohup bash save_svs_to_tiles.sh 1 4 &
nohup bash save_svs_to_tiles.sh 2 4 &
nohup bash save_svs_to_tiles.sh 3 4 &

exit 0
