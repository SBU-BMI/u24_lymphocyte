import os
from shutil import copyfile
import sys

src_dir = sys.argv[1];
des_dir = sys.argv[2];

src_list = [];
with open(src_dir + '/' + 'label.txt') as f:
    src_list = f.readlines();

des_list = [];
for i in range(30):
    des_list.append(src_list[i]);
    filename = src_list[i].split(' ')[0];
    src_file = src_dir + '/' + filename;
    des_file = des_dir + '/' + filename;
    copyfile(src_file, des_file);

f = open(des_dir + '/' + 'label.txt', 'w');
for id, line in enumerate(des_list):
    f.write(line);

f.close();
