# Survival Curve Pipline

Assumes that a file called `inputs` and `outputs` are in the same folder.

1. We first need to convert pathology images that have infiltrating lymphocytes identified (CNN outputs) into csv files (input of cluster indices).

You need to modify
a. gen_csv.sh
a.i. CNNOUTPUT: output of the CNN, the folder containing subfolders with the names rates-cancertype-all-auto.
a.ii. CSVFOLDER: any name for the folder of csv files
a.iii. OUTFOLDER: any name for the folder of results of cluster indices

b. populate_inputs.py
b.i. cancer_types: array of cancer types you want to process

    > `nohup bash gen_csv.sh >log.txt &`

    This may take long time to run.

Along with generated csv files, a master csv file named input_full.csv is created. This is the input of the next process

2. These csvs can be processed by an R script which runs spatial statistics on presense/absence data.
    > `nohup ./run_all.sh input_full.csv 6 > output.log &`

    Be aware that files consume varying amounts of memory and if memory is full, threads will fail.

3. `collateClusterIdx.sh` collects all the statistics into csv files.

4. These were then sent to MD Anderson for surival curve analysis.
