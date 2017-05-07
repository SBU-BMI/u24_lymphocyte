#!/bin/bash

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

# Scp all meta and heatmap files to osprey
bash scp_json.sh

# Put the heatmaps in the database
bash operation_on_remote_camic.sh "cd /home/lehhou/heatmap_pipeline; bash put_heatmaps_in_all.sh ${DEFAULT_TYPE}"

exit 0
