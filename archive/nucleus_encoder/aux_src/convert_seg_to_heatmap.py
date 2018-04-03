import numpy as np

img_file = '';
pred_file = '';

mu = 0;
sigma = 0;

PS = 200;
APS = 333;

# Load mu and sigma
mu_loaded, sigma_loaded, param_values = pickle.load(open(model_file, 'rb'));
mu = mu_loaded;
sigma = sigma_loaded;

print "Mu, Sigma = ", mu, sigma;

img = np.load(img_file);
prediction = np.load(pred_file);

pred_raw = np.copy(prediction).reshape(-1, PS, PS);

# Normalize and Reshape
img = img*sigma + mu;
img[img > 255] = 255;
img[img < 0] = 0;
img = img.astype(np.uint8);
