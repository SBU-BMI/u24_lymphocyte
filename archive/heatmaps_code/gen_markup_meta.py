from bson import json_util

slide_list_file = '/data08/shared/lehhou/necrosis_segmentation_workingdir/slide_list.txt';

# read from file
with open(slide_list_file) as f:
    content = f.readlines();
    #lines = content.split('\n');

print content[-1][-2];
#print (content[-1][-3]);
