from __future__ import print_function, division
import os


output_file = 'threshold_list.txt'
inputfile1 = './clicks/groups.txt'
inputfile2 = 'thresholds_group_user_defined.txt'
output = ''

def readTxtFile(filename):
    with open(filename, 'r') as f1:
        input1 = f1.readlines()
        input1 = [x.strip() for x in input1]
    return input1

input1 = readTxtFile(inputfile1)
input2 = readTxtFile(inputfile2)

thresholds_group = {}
for group in input2:
    line = group.split()
    thresholds_group[line[0]] = [line[1], line[2]]

for line in input1:
    line = line.split()
    thresholds = thresholds_group[line[-1]]
    temp = line[0] + ' ' + thresholds[0] + ' ' + thresholds[1] + '\n'
    output += temp

with open(output_file, 'w') as f:
    f.write(output)
