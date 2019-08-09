#/bin/bash

CNNOUTPUT='cnn_output'
CSVFOLDER="inputs"
OUTFOLDER="output"

mkdir $CSVFOLDER

nohup python populate_inputs.py $CNNOUTPUT $CSVFOLDER >log.txt 

mkdir $OUTFOLDER

ls -1 $CSVFOLDER/*/*.csv | awk -v outf="$OUTFOLDER" -F'/' '{print $0","outf"/"$2"/,"$2}' > input_full.csv
