# Instructions to run this stage

## 1. copy the folder data/patches\_from\_heatmaps to here "patches\_from\_heatmap".

## 2. Run step1.sh
    + to prepare the images for display on website
    + rename images' names to format 1.png, 2.png,...
    + add 100x100 rectangle to the center of the image

## 3. show the website to pathologist to select the positive samples
----this is gating step, must be completed before executing next steps-----

## 4. Run Step2.sh
to aggregate decisions from pathologists
inputs: clicked\_xxx.txt and ignored\_xxx.txt
outputs: 
### clicks/groups.txt contains the all slides ID and group it belongs to (1 of 7 groups named A,B,C,D,E,F, and G)
### groups\_sampling.txt contains maximum 8 slide ID per group
GIVE groups\_sampling.txt to pathologists. Pathologists will view each slide in this list on CAmicroscope, adjust the thresholds (lym specificity and Nec sensitivity) and record the thresholds for theat slide.

## 5. Must mannually create the file thresholds\_group\_user\_defined.txt that contains the thresholds for each group
the output is thresholds for each slide
From the thresholds recorded from step 4, compute the threshold for each group by taking the average of thresholds from slides in the same group.

-----this is gating step, must be completed before executing next steps--------

## 6. Run step3.sh
to generate thresholds for each slides by matching its group to group's threshold
inputs: groups.txt, thresholds\_group\_user\_defined.txt (hard coded in the code)
output: threshold\_list.txt

## 7. Copy file threshold\_list.txt to data/threshold\_list
