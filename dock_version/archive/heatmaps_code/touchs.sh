while read line; do
    touch read/`echo ${line} | awk -F'/' '{print $NF}'`
done < /data07/shared/lehhou/openslide_svs_extract_controller/distribute_svs_read.txt
