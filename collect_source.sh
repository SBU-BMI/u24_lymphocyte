#/bash/bin

FOLDERLIST=source_list.txt
OUT_FOLDER=`pwd`
SUFFIX="py m sh awk"
LIST_FILE_TMP=/tmp/cp_py_file_list.txt

while read subfol; do
    cd ${subfol}/
    FD_NAME=`echo ${subfol} | awk -F'/' '{if(length($NF)==0){print $(NF-1)}else{print $NF}}'`

    rm -rf ${LIST_FILE_TMP}
    for SUF in ${SUFFIX}; do
        find -name '*.'${SUF} >> ${LIST_FILE_TMP}
    done

    mkdir -p ${OUT_FOLDER}
    cd ${OUT_FOLDER}

    while read line; do
        SUB_NAME=`echo ${line} | awk -F'/' '{for(i=1;i<NF;++i){printf("%s/",$i)}}'`;
        mkdir -p ${FD_NAME}/${SUB_NAME};
        cp ${subfol}/${line} ${FD_NAME}/${SUB_NAME}/;
    done < ${LIST_FILE_TMP}

done < $FOLDERLIST


