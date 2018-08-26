#!/bin/bash

for i in `ls output`; do
	echo $i;
	featuredb-loader --dbhost quip3.bmi.stonybrook.edu --dbname quip --inptype csv --quip output/$i --fromdb
done
