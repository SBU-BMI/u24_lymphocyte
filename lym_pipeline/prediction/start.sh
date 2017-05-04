#!/bin/bash

rm -rf log.*.txt
nohup bash pred_thread.sh ../patches/ 0 2 > log.pred_thread_0.txt &
nohup bash pred_thread.sh ../patches/ 1 2 > log.pred_thread_1.txt &
nohup bash color_stats.sh ../patches/ 0 2 > log.color_stats_0.txt &
nohup bash color_stats.sh ../patches/ 1 2 > log.color_stats_1.txt &
wait

exit 0
