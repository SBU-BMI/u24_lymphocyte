#!/bin/bash
# get labeled patches
#   input: list_user.txt, list_case.txt, list_type.txt
#   output: ./patches/*

bash get_raw_human_annotations.sh
bash get_annotation_map.sh
bash extract_patches_from_annotation_maps.sh

exit 0
