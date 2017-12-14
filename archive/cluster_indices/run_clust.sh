#!/bin/bash

a=`echo $1 | awk -F ',' '{print $1}'`;
b=`echo $1 | awk -F ',' '{print $2}'`;
c=`echo $1 | awk -F ',' '{print $3}'`;

Rscript clust_index_single.R $PWD $a $b $c
