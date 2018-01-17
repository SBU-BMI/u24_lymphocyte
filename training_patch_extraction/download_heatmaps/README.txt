This module downloads modified heatmaps to png files under ./heatmaps/
Please check out get_modified_heatmap.sh for how to prepare input for it.

As an example, you can directly run get_modified_heatmap.sh
bash get_modified_heatmap.sh
and check the output under ./heatmaps/

The heatmap png files are color coded:
Red channel     0: marked lym negative
Red channel   255: marked lym positive
Red channel    64: predicted lym negative
Red channel   128: predicted lym positive
Green channel   0: marked tumor negative
Green channel 255: marked tumor positive

For each png file, there is a corresponding csv file that contains information
of all modified patches.
