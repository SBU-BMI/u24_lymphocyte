import sys
from necrosis_lib import necrosis_train

# If we keep training from some existing model, provide the path of the model file, otherwise, provide None
existing_model = None;

# List of training validation folders, which contain "image_*.png" and "mask_*.png" images
training_data_path = sys.argv[1];
dataset_list = training_data_path + '/nec_data_list.txt';
lines = [line.rstrip('\n') for line in open(dataset_list)];
validation_folder_list = [training_data_path + "/" + s for s in lines[0].split()];
training_folder_list = [training_data_path + "/" + s for s in lines[1].split()];

# The filename to store trained network while traning
save_model_path = sys.argv[2] + '/cnn_nec_model.pkl';

# Size of the loaded image and the input of the network
APS = 500;
PS = 200;

# Number of epoch everytime saving the model and testing on validation set
epochno_model_save = 50;
epochno_validate = 10;

necrosis_train(existing_model, save_model_path, training_folder_list, validation_folder_list, \
        APS, PS, epochno_model_save, epochno_validate);

