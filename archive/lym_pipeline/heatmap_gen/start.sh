#!/bin/bash

source ../conf/bashrc_base.sh
source ../conf/variables.sh

rm -rf json patch-level-lym patch-level-nec patch-level-merged
mkdir  json patch-level-lym patch-level-nec patch-level-merged

# Copy heatmap files from lym and necrosis prediction models
# to patch-level/ and necrosis/ folders respectively.
bash cp_heatmaps_all.sh

# Combine patch-level and necrosis heatmaps into one heatmap.
# Also generate high-res and low-res version.
bash combine_lym_necrosis_all.sh

# Generate meta and heatmap files for high-res and low-res heatmaps.
bash gen_all_json.sh

# Scp json files to osprey:/home/lehhou/heatmap_pipeline/json_todo/
#bash scp_json.sh

# Put empty humanmarks
#bash put_empty_humanmark.sh

exit 0
