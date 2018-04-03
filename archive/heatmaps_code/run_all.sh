#!/bin/bash

# Copy heatmap files from lym and necrosis prediction models
# to patch-level/ and necrosis/ folders respectively.
# Will not overwrite existing ones: only copy files that does not exist.
#bash cp_all_heatmaps.sh

# Combine patch-level and necrosis heatmaps into one heatmap.
# Also generate high-res and low-res version.
# Will not overwrite existing ones: only generate files that does not exist under gened/
bash combine_lym_necrosis_all.sh

# Generate meta and heatmap files for high-res and low-res heatmaps.
# Will not overwrite existing ones: only generate files that does not exist under gened/
bash gen_all_json.sh

# Scp all meta and heatmap files to osprey:/home/lehhou/heatmap/
# After executing this command, please go to osprey and run:
# [lehhou@osprey ~]$ cd /home/lehhou/heatmap
# [lehhou@osprey heatmap]$ bash put_heatmaps_in_all.sh
bash cp_json.sh

exit 0
