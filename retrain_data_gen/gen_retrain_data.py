from __future__ import print_function, division
import os
import glob
import numpy as np
import sys

# usage: the labels in txt file must be located in the source_folder
print("========Usage: python source_folder dest_folder==========")

source_folder = sys.argv[1]
dest_folder = sys.argv[2]

#source_folder = "../data/patches_from_heatmap"
#dest_folder = "./"

train_percentage = 0.8  # to split training and validation data
txt_files = [file for file in glob.glob(source_folder + "/*.txt")]

slides = []
all_rows = []
print(txt_files)
for file in txt_files:
    with open(file) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    all_rows = all_rows + lines

all_rows = np.array(all_rows)   # convert to np.array to do indexing
rands = np.random.permutation(len(all_rows))
trainset = list(all_rows[rands[:np.int32(train_percentage*len(rands))]])
valset = list(all_rows[rands[np.int32(train_percentage*len(rands)):]])

for row in all_rows:
    line = row.split()
    if line[-1] not in slides:
        slides.append(line[-1])
for slide in slides:    # delete folders if exists
    os.system("rm -rf " + dest_folder + "/" + slide + "*")

def MKDIR(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_data(dataset, folder = "trains"):
    i = 0
    for line in dataset:
        index = i//5000 # subfolder, each folder contains max 5000 images
        row = line.split()
        vals = dest_folder + "/" + row[-1] + "_" + folder + "_" + str(index+1)
        val_label = vals + "/" + "label.txt"
        MKDIR(vals)
        img_name = str(i) + ".png"
        os.system("cp " + source_folder + "/" + row[0] + " " + vals + "/" + img_name)
        with open(val_label, 'a') as f:
            f.write(img_name + " " + row[1] + " " + row[0] + " " + row[2] + "\n")
        i += 1

load_data(trainset, folder = "trains")
load_data(valset, folder = "vals")
