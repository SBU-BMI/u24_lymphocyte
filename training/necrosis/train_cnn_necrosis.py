from necrosis_lib import necrosis_train

def main():
    # If we keep training from some existing model, provide the path of the model file, otherwise, provide None
    existing_model = None;

    # The filename to store trained network while traning (the suffix "_n.pkl", where n is the epoch number, will be automatically added)
    save_model_path = './models/cnn_model_mu_sigma_necrosis';

    # List of training validation folders, which contain "image_*.png" and "mask_*.png" images
    training_folder_list = ['../data/small_cnn_necrosis_train_data'];
    validation_folder_list = ['../data/small_cnn_necrosis_val_data'];

    # Size of the loaded image and the input of the network
    APS = 500;
    PS = 200;

    # Number of epoch everytime saving the model and testing on validation set
    epochno_model_save = 50;
    epochno_validate = 10;

    necrosis_train(existing_model, save_model_path, training_folder_list, validation_folder_list, APS, PS,   epochno_model_save, epochno_validate);

if __name__ == '__main__':
    main();

