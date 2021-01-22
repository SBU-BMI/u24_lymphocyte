# Instructions for running the TIL Pipeline

The TIL Pipeline *[Tumor-Infiltrating Lymphocytes (TIL)]* has four phases:

1. **Prediction phase**
	* Run `svs_2_heatmap.sh`, optionally comment out upload\_heatmaps.sh in heatmap\_gen/start.sh
	    * Tiles the svs image into PNGs and saves them to data/patches
	    * Runs prediction on the patches, and saves the data as text files to data/patches
	        * patch-level-color.txt
	        * patch-level-lym.txt
	        * patch-level-necrosis.txt
	    * Generates heatmap data as JSON files in heatmap\_jsons, and corresponding prediction data as text files heatmap\_txt
	
2. **Upload/review/refine**
	* Upload data `(upload_heatmaps.sh)` & ask pathologists to review and refine the visualized data.
	
3. **Retraining dataset generation phase**
	* Run code in `download_heatmap`; specifically, download\_training\_patches.sh and/or download\_training\_tumor\_patches.sh
	* It fetches the human-generated markups, generates heatmap weights, and saves the data as text in raw\_marking\_xy
	* Generates modified heatmaps from the weight information and stores the data as a csv file in modified\_heatmaps, along with a visualization stored as PNG
	* Writes training patches as PNGs to patches\_from\_heatmap
	
    A. **Manual Step**
    
    * Create a new folder in training\_data\_cnn
    * Copy patches\_from\_heatmap/* to training\_data\_cnn/[new\_folder]
    * Append [new\_folder] to the end of file lym\_data\_list.txt
    
4. **Generate new cnn model**
	* Run `train_models.sh`; specifically, training/lymphocyte/start\_cnn\_lymphocyte\_training.sh
	* Generates a cnn\_lym\_model.pkl file in models\_cnn
	
## Setup
During the retraining phase, download\_markings\_weights.sh reads caseids from raw\_marking\_to\_download\_case\_list/case\_list.txt, so be sure to update the file when you're ready.

Modify conf/variables.sh:

* MONGODB\_HOST
* MONGODB\_PORT
* BASE\_DIR
* USERNAME
* HEATMAP\_VERSION

**Note:** Whenever we update our CNN model, we need to assign a new execution id for that. This is a manual step, atm.

Put image file(s) in data/svs.

Install theano in home directory.
