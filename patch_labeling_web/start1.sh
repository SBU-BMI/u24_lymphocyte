#!bin/bash

# start first to aggregate the clicks into ./clicks/groups.txt
# then generate maximum 8 slides per group, stored at store at ./clicks/groups_sampling.txt

bash clicks_summarize.sh
wait;

python -u groups_gen.py     # output is maximum 8 slides per group, store at ./clicks/groups_sampling.txt
wait;

exit 0
