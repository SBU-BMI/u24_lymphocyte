# Instructions to run this stage

## 1. copy the folder data/patches\_from\_heatmaps to here "patches\_from\_heatmaps".

## 2. Run step1.sh
    + to prepare the images for display on website
    + rename images' names to format 1.png, 2.png,...
    + add 100x100 rectangle to the center of the image

## 3. show the website to pathologist to select the positive samples
----this is gating step, must be completed before executing next steps-----

## 4. Run Step2.sh
to aggregate decisions from pathologists
inputs: clicked\_xxx.txt and ignored\_xxx.txt
outputs: groups.txt contains the slides belong to 1 of 7 groups A,B,C,D,E,F, and G

## 5. Must mannually create the file thresholds\_group\_user\_defined.txt that contains the thresholds for each group
the output is thresholds for each slide
there are maximum 8 slides for each group chosen. User must mannually go through each slide, adjust the threshold and record them down. Take average of 8 threshold to be the threshold for that group.

-----this is gating step, must be completed before executing next steps--------

## 6. Run step3.sh
to generate thresholds for each slides by matching its group to group's threshold
inputs: groups.txt, thresholds\_group\_user\_defined.txt (hard coded in the code)
output: threshold\_list.txt

## 7. Copy file threshold\_list.txt to data/threshold\_list
