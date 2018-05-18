from __future__ import print_function, division
import os
import json

groups_file = "./clicks/groups.txt"
output_file = "./clicks/groups_sampling.txt"
N = 8       # maximum number of slides per group
NoGroups = 7

groups = [] # to save all possible groups
output = {} # save the output
count = 0   # keep track of total slides saved

with open(groups_file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
for line in content:
    line = line.split()
    group = line[-1]
    if group not in groups:
        groups.append(group)
        output[group] = [line[0]]
    else:
        if len(output[group]) < N:
            output[group].append(line[0])
            count += 1
    if count > N*NoGroups:
        break

with open(output_file, 'w') as f:
    f.write(json.dumps(output))

