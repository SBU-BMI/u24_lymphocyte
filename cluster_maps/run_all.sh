#!/bin/bash

for i in `ls cluster_csv`; do
    echo $i;
    python ./quip_csv.py cluster_csv/$i;
done 
