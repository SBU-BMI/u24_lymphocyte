import os
import sys
import pickle

from pred_necrosis_wsi import predict_slide

slide_path = sys.argv[1];
classification_model_file = sys.argv[2] + '/cnn_nec_model.pkl';
classification_model_file = '/data01/shared/hanle/results/necrosis_5_0.9684.pkl'
heat_map_out = sys.argv[3];

def load_model_value(model_file):
    loaded_var = pickle.load(open(model_file, 'rb'));
    mu = loaded_var[0];
    sigma = loaded_var[1];

    param_values = loaded_var[2];
    return mu, sigma, param_values;


def Segment_Necrosis_WholeSlide():
    mu, sigma, param_values = load_model_value(classification_model_file);
    output_path = slide_path;
    predict_slide(slide_path, mu, sigma, param_values, output_path, heat_map_out);


if __name__ == "__main__":
    Segment_Necrosis_WholeSlide();


