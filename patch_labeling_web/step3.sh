#!bin/bash

# Run this step after start1.sh
# must mannually create the file thresholds_group.txt that contains the thresholds for each group
# the output is thresholds for each slide
# basically aggregate the slides from ./clicks/groups.txt and thresholds_group.txt


python -u thresholds_group_gen.py

exit 0
