import pickle
import scipy.misc
import numpy as np
import os
import PIL
import glob

from PIL import Image
from pred_necrosis_batch import necrosis_predict


def load_seg_data(train_folder_list, test_folder_list, APS):
    X_train = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_train = np.zeros(shape=(0, APS, APS), dtype=np.float32);
    X_test = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_test = np.zeros(shape=(0, APS, APS), dtype=np.float32);
    image_name_train = [];
    image_name_test = [];

    for train_set in train_folder_list:
        X_tr, y_tr, image_name_train = load_seg_data_folder(train_set, APS);
        X_train = np.concatenate((X_train, X_tr));
        y_train = np.concatenate((y_train, y_tr));

    for test_set in test_folder_list:
        X_ts, y_ts, image_name_test = load_seg_data_folder(test_set, APS);
        X_test = np.concatenate((X_test, X_ts));
        y_test = np.concatenate((y_test, y_ts));

    print "Shapes: ", X_train.shape, X_test.shape;
    return X_train, y_train.astype(np.int32), image_name_train, X_test, y_test.astype(np.int32), image_name_test;


def load_seg_data_folder(folder, APS):
    X = np.zeros(shape=(40000, 3, APS, APS), dtype=np.float32);
    y = np.zeros(shape=(40000, APS, APS), dtype=np.float32);

    idx = 0;
    image_names = read_image_list_file(folder + '/list.txt');
    for img_name in image_names:
        # Load file
        loaded_png = Image.open(folder + '/' + img_name + '.png');
        resized_png = loaded_png.resize((APS, APS), PIL.Image.ANTIALIAS);
        img_png = np.array(resized_png.convert('RGB')).transpose();
        mask_png = np.zeros(shape=(1, APS, APS), dtype=np.float32);
        X[idx] = img_png;
        y[idx] = mask_png;
        idx += 1;

    X = X[:idx];
    y = y[:idx];

    return X, y, image_names;

def read_image_list_file(text_file_path):
    with open(text_file_path) as f:
        content = f.readlines();

    content = [x.strip() for x in content];
    return content;

def get_img_idx(folder, prefix='image_'):
    file_idx = np.zeros(shape=(40000,), dtype=np.int32);
    id = 0;
    print folder + '/' + prefix + '*.png';
    for filename in glob.glob(folder + '/' + prefix + '*.png'):
        file_no_part = filename[(filename.rfind('_') + 1):];
        file_idx[id] = int(file_no_part[:-4]);
        id += 1;

    file_idx = file_idx[:id];
    file_idx = np.sort(file_idx);
    return file_idx;

def predict_slide(slide_folder, mu, sigma, param_values, output_path, heat_map_out):
    # Get list of image files
    list_file_path = slide_folder + '/list.txt';
    img_name_list = [];
    if (os.path.isfile(list_file_path) == False):
        print "list file not avaible, producing a list file";
        f = open(list_file_path, 'w')
        path_list = glob.glob(slide_folder + '/*.png');
        for img_path in path_list:
            base=os.path.basename(img_path);
            img_name = os.path.splitext(base)[0];
            f.write(img_name + '\n');
        f.close();

    with open(list_file_path) as f:
        content = f.readlines();
    img_name_list = [x.strip() for x in content];

    # Analyze APS, PS
    APS = 333;
    PS = 200;

    # Load testing data
    print ('Load testing data...');
    X_train, y_train, image_name_train, X_test, y_test, image_name_test = load_seg_data([], [slide_folder], APS);
    print ('Finish loading testing data');

    # Do prediction
    print ('Do prediction...');
    image_array, groundtruth_array, prediction_array = necrosis_predict(X_test, y_test, mu, sigma, param_values, APS, PS);
    print "Output shape: image, groundtruth, prediction ", image_array.shape, groundtruth_array.shape, prediction_array.shape;

    parent_path, slide_name = os.path.split(slide_folder);
    heatmap_path = output_path + '/' + heat_map_out;
    f_res = open(heatmap_path, 'w')
    for idx, big_patch_name in enumerate(image_name_test):
        parts = big_patch_name.split('_');
        root_x = int(parts[0]);
        root_y = int(parts[1]);
        abs_size = int(parts[2]);
        big_patch = prediction_array[idx];

        loc_arr = [x * 0.1 + 0.05 for x in range(0, 10)];

        for x_idx, abs_x in enumerate(xrange(0,300,33)):
            for y_idx, abs_y in enumerate(xrange(0,300,33)):
                real_x_loc = int(loc_arr[x_idx] * abs_size + root_x);
                real_y_loc = int(loc_arr[y_idx] * abs_size + root_y);
                avg_val = np.average(big_patch[abs_x : abs_x + 33, abs_y : abs_y + 33]);
                f_res.write("{0} {1} {2}\n".format(real_x_loc, real_y_loc, avg_val));

    f_res.close();

