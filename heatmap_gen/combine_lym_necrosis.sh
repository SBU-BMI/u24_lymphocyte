#!/bin/bash
# merges two heatmaps
# usage:
#   bash merge_lym_necrosis.sh ${FILE}
#   ${FILE}: something like prediction-TCGA-44-2655-01Z-00-DX1.ee255271-780c-461c-ab23-5cd3504b5e4a

FN=$1
LYM_FOLDER=./patch-level-lym/
NEC_FOLDER=./patch-level-nec/
OUT_FOLDER=./patch-level-merged/

awk 'NR==FNR{
    x=$1;
    y=$2;
    h[x" "y]=$3;
    if(x_before!=x){
        lower[x]=x_before;
        higher[x_before]=x;
    }
    if(y_before!=y){
        lower[y]=y_before;
        higher[y_before]=y;
    }
    x_before=x;
    y_before=y;
}

NR!=FNR{
    x=-1;
    y=-1;
    for (i=$1-2;i<=$1+2;++i) {
        for (j=$2-2;j<=$2+2;++j) {
            if (i" "j in h) {
                x = i;
                y = j;
            }
        }
    }
    lym = $3;

    if (x" "y in h) {
        necrosis = h[x" "y];

        if (h[lower[x]" "y] > necrosis)
            necrosis = h[lower[x]" "y];
        if (h[higher[x]" "y] > necrosis)
            necrosis = h[higher[x]" "y];
        if (h[x" "lower[y]] > necrosis)
            necrosis = h[x" "lower[y]];
        if (h[x" "higher[y]] > necrosis)
            necrosis = h[x" "higher[y]];

        if (necrosis > 0.4) {
            combo = 0.01;
        } else {
            combo = lym;
        }
    } else {
        combo = lym;
        necrosis = 0;
    }

    if (length(necrosis) == 0) {
        necrosis = 0;
    }
    print $1, $2, lym, necrosis;
}' ./${NEC_FOLDER}/${FN} ./${LYM_FOLDER}/${1} > ${OUT_FOLDER}/${FN}

exit 0
