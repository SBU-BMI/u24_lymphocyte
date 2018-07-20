# Instructions for running the heatmap pipeline

## Setup

### Config File
Modify conf/variables.sh

* MONGODB\_HOST
* MONGODB\_PORT
* BASE\_DIR
* USERNAME
* Install theano in home directory

### SVS
Put image file(s) in data/svs.


### Output
We expect output in the following directories, so clear the contents of these dirs from previous runs if necessary:

* heatmap\_jsons
* heatmap\_txt
* log
* patches


### List of Case IDs
Some parts of the pipeline read from this file:
raw\_marking\_to\_download\_case\_list/case\_list.txt


### CNNs
* models\_cae/cae\_model.pkl
* models\_cnn/cnn\_lym\_model.pkl
* models\_cnn/cnn\_nec\_model.pkl


## Usage
Run:

```
scripts/svs_2_heatmap.sh
```

1. Extracts patches for analysis
2. Performs the prediction 
3. Generates heatmaps (JSON files)
