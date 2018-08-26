#!/bin/bash

for i in `cat rdata.list `; do 
    a=`echo $i | awk -F ',' '{print $2}'`;
    b=`echo $i | awk -F ',' '{print $1}'`;
	Rscript extract_clusters.R $a $b; 
done
