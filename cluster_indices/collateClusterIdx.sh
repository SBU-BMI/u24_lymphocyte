#!/bin/bash

# This script collates indices into one CSV file so they may be interpreted.

# If running collateClusterIdx.sh standalone, uncomment removal:
#rm ./output/*totalClusterStats.csv  # remove previous populates

FOLDERS=./output/*
topflag=1
for f in $FOLDERS
    do
    #append file name
    outputName=$(echo "./output/"$(echo $f | rev | cut -d '/' -f -1 | rev)"_totalClusterStats.csv")
    echo "outputName:" $outputName
    touch $outputName

    for ff in $(find $f/*_clusterInfo.csv -maxdepth 1)
        do
        #open ClusterInfo and read out first two lines into variables
        line1=$(awk 'FNR==1{print $0}' $ff)
        line2=$(awk 'FNR==2{print $0}' $ff)
        #filename=$(echo $ff | awk -F '/' '{print $4}' | rev | cut -d '_' -f 3 | rev)
        temp=$(awk -F "," 'FNR==2{print $1}' $ff)
        filename=$(echo "$f/"$(echo "${temp:1:${#temp}-2}")"_indices_ap.csv")

        #append first line if its on top
        if [ "$topflag" == "1" ]; then
            line1=$(echo "$line1"","$(awk 'FNR==1{print $0}' $filename))
            echo $line1 >> $outputName
        fi

        #otherwise always append second line
        line2=$(echo "$line2"","$(awk 'FNR==2{print $0}' $filename))
        echo $line2 >> $outputName
        topflag=0
    done

    topflag=1

done
