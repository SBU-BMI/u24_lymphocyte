#/bash/bin

FOLDERLIST=source_list.txt
OUT_FOLDER=`pwd`
SUFFIX="py m sh awk"
LIST_FILE_TMP=/tmp/cp_py_file_list.txt

while read fol_setting; do
    echo processing ${fol_setting}

    subfol=`echo ${fol_setting} | awk '{print $1}'`
    max_dp=`echo ${fol_setting} | awk '{print $2}'`

    cd ${subfol}/
    FD_NAME=`echo ${subfol} | awk -F'/' '{if(length($NF)==0){print $(NF-1)}else{print $NF}}'`

    rm -rf ${LIST_FILE_TMP}
    for SUF in ${SUFFIX}; do
        if [ "${max_dp}" == "" ]; then
            find -name '*.'${SUF} -type f >> ${LIST_FILE_TMP}
        else
            find -name '*.'${SUF} -type f -maxdepth ${max_dp} >> ${LIST_FILE_TMP}
        fi
    done

    mkdir -p ${OUT_FOLDER}
    cd ${OUT_FOLDER}

    while read line; do
        SUB_NAME=`echo ${line} | awk -F'/' '{for(i=1;i<NF;++i){printf("%s/",$i)}}'`;
        mkdir -p ${FD_NAME}/${SUB_NAME};
        cp ${subfol}/${line} ${FD_NAME}/${SUB_NAME}/;
    done < ${LIST_FILE_TMP}

done < $FOLDERLIST
