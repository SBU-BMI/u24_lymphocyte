import numpy as np
import os
import sys
import cv2
from PIL import Image

preds_fol = '../data/heatmap_txt'
labels_fol = '../data/tumor_labeled_heatmaps'

preds_fol = sys.argv[1]
labels_fol = sys.argv[2]

print('Usage: python compute_dice_score.py prediction-folder labeled-image-folder')

resize = 25
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
    if 'prediction' in p and 'low_res' not in p:
        preds.append(p)

print('prediction files: ', preds)

def check_label(labels, svs_id):
    for l in labels:
        if svs_id in l:
            return l
    return None

for pred in preds:
    svs_id = pred.split('prediction-')[-1]
    label = check_label(labels, svs_id)
    if label is not None:
        png = cv2.imread(os.path.join(labels_fol, label), 0)
        png = cv2.resize(png, (0, 0), fx = 1/resize, fy = 1/resize)
        png[png < 230] = 0
        png = png/np.max(png)
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
        for p in predictions:
            p = p.split()
            y, x, score = int(int(p[0])/(4*resize)), int(int(p[1])/(4*resize)), float(p[2])
            if score > 0.0:
                # update pred_M:
                pred_M[x : x + patch_size, y : y + patch_size] = int(score > threshold)

        A = pred_M * png
        B = png - A
        C = pred_M - A
        D = np.ones((h, w), dtype = np.float32) - A - B - C
        D[D < 0] = 0

        sA = np.sum(A)
        sB = np.sum(B)
        sC = np.sum(C)
        sD = np.sum(D)
        print('Done processed ', pred)
        print('Accuracy: ', (sA + sD)/(sA + sB + sC + sD))
        print('Dice-score/F1-score: ', 2*sA/(2*sA + sB + sC))











