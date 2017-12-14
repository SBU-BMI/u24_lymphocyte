# Survival Curve Pipline

Assumes that a file called `inputs` and `outputs` are in the same folder.

1. We start with pathology images that have infiltrating lymphocytes identified (CNN outputs). In this case these files were found in `/data08/shared/lehhou/active_learning_osprey`. There's a script that processes these files into csvs that contain presence/absence locations of limphocytes on slide. To obtain these csv files, run:
    > `nohup matlab -nodisplay -r "run populate_inputs.m; exit" </dev/null &>log.txt &`

    This will run for a day or so.

2. These csvs can be processed by an R script which runs spatial statistics on presense/absence data.
    > `nohup ./run_all.sh input_full.csv 6 > output.log &`

    Be aware that files consume varying amounts of memory and if memory is full, threads will fail.

3. `collateClusterIdx.sh` collects all the statistics into csv files.

4. These were then sent to MD Anderson for surival curve analysis.
