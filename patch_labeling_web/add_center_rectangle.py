import cv2
import os


s = 200
l = 100

files = os.listdir('images/')
for file in files:
    if '.png' not in file: continue
    fn = 'images/' + file

    im = cv2.imread(fn)
    im[s:s + l, s:s+3, 2] = 255
    im[s:s+3, s: s + l, 2] = 255
    im[s: s + l, s + l: s + l + 3, 2] = 255
    im[s + l: s + l + 3, s: s + l, 2] = 255

    im[s:s + l, s:s+3, 0:2] = 0
    im[s:s+3, s: s + l, 0:2] = 0
    im[s: s + l, s + l: s + l + 3, 0:2] = 0
    im[s + l: s + l + 3, s: s + l, 0:2] = 0

    cv2.imwrite(fn, im)