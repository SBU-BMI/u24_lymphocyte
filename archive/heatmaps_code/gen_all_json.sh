#!/bin/bash

mv ../heatmaps_v3/*.json ../heatmaps_v3/backup/
VER=lym_v8

for files in prediction-*; do
    if [ ! -f gened/${files} ]; then
        cp ${files} ../heatmaps_v3/
    fi
    mv ${files} gened/
done

cd ../heatmaps_v3/
for files in prediction-*; do
    if [[ "$files" == *.low_res ]]; then
        if   [ -f /data01/tcga_data/tumor/luad/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  luad /data01/tcga_data/tumor/luad/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  luad /data01/tcga_data/tumor/luad/ lym 0.5 necrosis 0.5
        elif [ -f /data03/tcga_data/tumor/brca/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  brca /data03/tcga_data/tumor/brca/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  brca /data03/tcga_data/tumor/brca/ lym 0.5 necrosis 0.5
        elif [ -f /data06/tcga_data/tumor/paad/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  paad /data06/tcga_data/tumor/paad/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  paad /data06/tcga_data/tumor/paad/ lym 0.5 necrosis 0.5
        elif [ -f /data08/tcga_data/tumor/uvm/`echo ${files}  | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  uvm  /data08/tcga_data/tumor/uvm/  lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  uvm  /data08/tcga_data/tumor/uvm/  lym 0.5 necrosis 0.5
        elif [ -f /data01/tcga_data/tumor/skcm/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  skcm /data01/tcga_data/tumor/skcm/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  skcm /data01/tcga_data/tumor/skcm/ lym 0.5 necrosis 0.5
        elif [ -f ./coad/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  coad /data07/tcga_data/tumor/coad/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  coad /data07/tcga_data/tumor/coad/ lym 0.5 necrosis 0.5
        elif [ -f ./read/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  read /data08/shared/lehhou/tcga/tumor/read/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-low_res  read /data08/shared/lehhou/tcga/tumor/read/ lym 0.5 necrosis 0.5
        else
            SVS=`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs
            TYPE_N_FOLDER=`bash get_type_n_folder.sh ${SVS}`
            TYPE=`echo ${TYPE_N_FOLDER} | awk '{print $1}'`
            FOLDER=`echo ${TYPE_N_FOLDER} | awk '{print $2}'`
            if [ ${TYPE} == "notfound" ]; then echo ${SVS} not found; continue; fi
            python gen_json_multipleheat_v3.py ${files} ${VER}-low_res ${TYPE} ${FOLDER} lym 0.5 necrosis 0.5
        fi
    else
        if   [ -f /data01/tcga_data/tumor/luad/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-high_res luad /data01/tcga_data/tumor/luad/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-high_res luad /data01/tcga_data/tumor/luad/ lym 0.5 necrosis 0.5
        elif [ -f /data03/tcga_data/tumor/brca/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-high_res brca /data03/tcga_data/tumor/brca/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-high_res brca /data03/tcga_data/tumor/brca/ lym 0.5 necrosis 0.5
        elif [ -f /data06/tcga_data/tumor/paad/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-high_res paad /data06/tcga_data/tumor/paad/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-high_res paad /data06/tcga_data/tumor/paad/ lym 0.5 necrosis 0.5
        elif [ -f /data08/tcga_data/tumor/uvm/`echo ${files}  | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-high_res uvm  /data08/tcga_data/tumor/uvm/  lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-high_res uvm  /data08/tcga_data/tumor/uvm/  lym 0.5 necrosis 0.5
        elif [ -f /data01/tcga_data/tumor/skcm/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-high_res skcm /data01/tcga_data/tumor/skcm/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-high_res skcm /data01/tcga_data/tumor/skcm/ lym 0.5 necrosis 0.5
        elif [ -f ./coad/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-high_res coad /data07/tcga_data/tumor/coad/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-high_res coad /data07/tcga_data/tumor/coad/ lym 0.5 necrosis 0.5
        elif [ -f ./read/`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs ]; then
            echo python gen_json_multipleheat_v3.py ${files} ${VER}-high_res read /data08/shared/lehhou/tcga/tumor/read/ lym 0.5 necrosis 0.5
            python gen_json_multipleheat_v3.py ${files} ${VER}-high_res read /data08/shared/lehhou/tcga/tumor/read/ lym 0.5 necrosis 0.5
        else
            SVS=`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs
            TYPE_N_FOLDER=`bash get_type_n_folder.sh ${SVS}`
            TYPE=`echo ${TYPE_N_FOLDER} | awk '{print $1}'`
            FOLDER=`echo ${TYPE_N_FOLDER} | awk '{print $2}'`
            if [ ${TYPE} == "notfound" ]; then echo ${SVS} not found; continue; fi
            python gen_json_multipleheat_v3.py ${files} ${VER}-high_res ${TYPE} ${FOLDER} lym 0.5 necrosis 0.5
        fi
    fi
    mv ${files} ./backup/
done

exit 0
