import numpy as np
import os
import sys
import cv2
from PIL import Image
#from skimage.morphology import square, dilation

preds_fol = '../data/heatmap_txt'
labels_fol = '../data/tumor_labeled_heatmaps'

preds_fol = sys.argv[1]
labels_fol = sys.argv[2]

print('Usage: python compute_dice_score.py prediction-folder labeled-image-folder')

resize = 1
threshold = 0.5

labels_temp = os.listdir(labels_fol)
labels = []
for l in labels_temp:
    if '.png' in l:
        labels.append(l)

print('labels: ', labels)

preds_temp = os.listdir(preds_fol)
preds = []
for p in preds_temp:
    if 'prediction' in p and 'intersected' not in p:
        preds.append(p)

print('prediction files: ', preds)

def check_label(labels, svs_id):
    for l in labels:
        if svs_id in l:
            return l
    return None

for pred in preds:
    low_res = False
    svs_id = pred.split('prediction-')[-1]
    if 'low_res' in svs_id:
        svs_id = svs_id.split('low_res')[0]
        low_res = True

    label = check_label(labels, svs_id)
    if label is not None:
        png = cv2.imread(os.path.join(labels_fol, label), 0)
        png = cv2.resize(png, (0, 0), fx = 1/resize, fy = 1/resize)
        png[png < 230] = 0
        png = png/np.max(png)
        # need to dilute to widen the tumor region by 5um ~ 20px in 40X ~ 5px in labeled-heatmap
        #png = png.astype(np.uint8)
        #png = dilation(png, square(5*2))

        #print('png: ', png)
        #print('size of ', label, ' :', png.shape)

        with open(os.path.join(preds_fol, pred)) as f:
            temp = f.readlines()

        predictions = [t.strip() for t in temp]
        if len(predictions) < 10: continue
        patch_size = 1 + int(np.abs(int(predictions[0].split()[1]) - int(predictions[1].split()[1]))/(4*resize))
        #print('patch_size: ', patch_size)
        h, w = png.shape

        pred_M = np.zeros((h, w), dtype = np.float32)
        #print('size of pred_M: ', pred_M.shape)

        f_intersect = open(os.path.join(preds_fol, pred + '.intersected'), 'w')
        for p in predictions:
            q = p.split()
            y, x = int(int(q[0])/(4*resize)), int(int(q[1])/(4*resize))
            if np.sum(png[x : x + patch_size, y : y + patch_size])/(patch_size*patch_size) > 0.5:   # the intersect is more than 50% --> take it
                f_intersect.write(p + '\n')
            else:
                f_intersect.write(q[0] + ' ' + q[1] + ' 0.0 ' + q[3] + '\n')

        f_intersect.close()
