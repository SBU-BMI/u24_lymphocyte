# Instructions for running the TIL Pipeline


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


### Models
* models\_cnn/cnn\_lym\_model.pkl
	* regions identified as containing lymphocytes
* models\_cnn/cnn\_nec\_model.pkl 
	* necrotic regions
* models\_cae/cae\_model.pkl 
	* auto encoded regions


## Usage

The TIL Pipeline *[Tumor-Infiltrating Lymphocytes (TIL)]* has four phases:

1. **Prediction phase**
	* Run `svs_2_heatmap.sh`, optionally comment out upload\_heatmaps.sh in heatmap\_gen/start.sh.
2. **Upload/review/refine**
	* Upload data `(upload_heatmaps.sh)` & ask pathologists to review and refine the visualized data.
3. **Retraining dataset generation phase**
	* Run code in `download_heatmap`; specifically, download\_training\_patches.sh and/or download\_training\_tumor\_patches.sh
4. **Generate new cnn model**
	* Run `train_models.sh`; specifically, start\_cnn\_lymphocyte\_training.sh