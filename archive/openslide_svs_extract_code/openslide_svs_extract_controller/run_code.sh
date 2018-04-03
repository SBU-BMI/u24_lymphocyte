#!/bin/bash

ssh nfs001 "nohup bash /data/shared/lehhou/openslide_svs_extract/main.sh &> /data/shared/lehhou/openslide_svs_extract/log.txt &"
ssh nfs002 "nohup bash /data/shared/lehhou/openslide_svs_extract/main.sh &> /data/shared/lehhou/openslide_svs_extract/log.txt &"
ssh nfs003 "nohup bash /data/shared/lehhou/openslide_svs_extract/main.sh &> /data/shared/lehhou/openslide_svs_extract/log.txt &"
ssh nfs004 "nohup bash /data/shared/lehhou/openslide_svs_extract/main.sh &> /data/shared/lehhou/openslide_svs_extract/log.txt &"
ssh nfs005 "nohup bash /data/shared/lehhou/openslide_svs_extract/main.sh &> /data/shared/lehhou/openslide_svs_extract/log.txt &"
ssh nfs006 "nohup bash /data/shared/lehhou/openslide_svs_extract/main.sh &> /data/shared/lehhou/openslide_svs_extract/log.txt &"
ssh nfs007 "nohup bash /data/shared/lehhou/openslide_svs_extract/main.sh &> /data/shared/lehhou/openslide_svs_extract/log.txt &"
ssh nfs008 "nohup bash /data/shared/lehhou/openslide_svs_extract/main.sh &> /data/shared/lehhou/openslide_svs_extract/log.txt &"
ssh nfs010 "nohup bash /data/shared/lehhou/openslide_svs_extract/main.sh &> /data/shared/lehhou/openslide_svs_extract/log.txt &"

exit 0
