#!/bin/bash

rm -rf json patch-level-lym patch-level-nec patch-level-merged
mkdir  json patch-level-lym patch-level-nec patch-level-merged

# Copy heatmap files from lym and necrosis prediction models
# to patch-level/ and necrosis/ folders respectively.
bash cp_heatmaps_all.sh &> ../log/log.cp_heatmaps_all.txt

# Combine patch-level and necrosis heatmaps into one heatmap.
# Also generate high-res and low-res version.
bash combine_lym_necrosis_all.sh &> ../log/log.combine_lym_necrosis_all.txt

# Generate meta and heatmap files for high-res and low-res heatmaps.
bash gen_all_json.sh &> ../log/log.gen_all_json.txt

# Put all jsons to camicroscope
bash upload_heatmaps.sh &> ../log/log.upload_heatmaps.txt

exit 0
