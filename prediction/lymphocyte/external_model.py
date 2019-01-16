import numpy as np

def load_external_model(model_path):
    # Load your model here
    model = load_model(model_path)
    return model

def pred_by_external_model(model, inputs):
    # Get prediction here
    # model:
    #     A model loaded by load_external_model
    # inputs :
    #     float32 numpy array with shape N x 3 x 100 x 100
    #     Range of value: 0.0 ~ 255.0
    #     You may need to rearrange inputs:
    #     inputs = inputs.transpose((0, 2, 3, 1))
    # Expected output:
    #     float32 numpy array with shape N x 1
    #     Each entry is a probability ranges 0.0 ~ 1.0
    pred = model.predict(inputs)
    return pred

