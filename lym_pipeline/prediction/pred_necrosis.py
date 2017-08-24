import os
import sys
import pickle

from pred_necrosis_wsi import predict_slide

classification_model_file = 'models/cnn_model_mu_sigma_necrosis.pkl';
slide_path = sys.argv[1];

def load_model_value(model_file):
    loaded_var = pickle.load(open(model_file, 'rb'));
    mu = loaded_var[0];
    sigma = loaded_var[1];

    param_values = loaded_var[2];
    return mu, sigma, param_values;


def Segment_Necrosis_WholeSlide():
    mu, sigma, param_values = load_model_value(classification_model_file);
    output_path = slide_path;
    predict_slide(slide_path, mu, sigma, param_values, output_path);


if __name__ == "__main__":
    Segment_Necrosis_WholeSlide();


