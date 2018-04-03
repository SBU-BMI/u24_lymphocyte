while read line; do
    if [ "`echo ${line} | awk '{print NF}'`" -eq 1 ]; then
        wc -l ${line}/label.txt
    else
        list=`echo ${line} | awk '{printf("%s",$1); for(i=2;i<=NF;++i){printf(",%s",$i);}}'`
        echo wc -l {${list}}/label.txt
    fi
done < ${1}
