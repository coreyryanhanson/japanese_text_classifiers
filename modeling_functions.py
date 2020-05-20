import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score, balanced_accuracy_score

from random_lumberjacks.src.random_lumberjacks.model.model_classes import *

def color_values_to_float(array, bits):
    """Converts arrays with integer color data to floats depending on their bit depth. Currently only supports
    8bit, 10 bit, and 12 bit images."""

    bit_dict = {8:255, 10:1023, 12:4095}
    return array/bit_dict[bits]

def determine_trainable_layers(model):
    for layer in model.layers:
        print(layer.name, layer.trainable)
    print(f"{len(model.trainable_weights)} trainable weights")

def images_to_1d(data, w=None, h=None, channels=None, inverse=False):
    """Function that will turn conform pixel data to their vector representation in order to be used in Neural Networks.
    Can be done in reverse, but then requires values for the width/height and number of color channels if they exist."""

    # Makes a copy of the data and counts the observations
    data, n_obs = data.copy(), data.shape[0]

    # The arguments passed into numpy's reshape function begin by assuming unvectorization with one color channel.
    args = [n_obs, w, h]

    # If there are color channels the arguments add them in.
    if inverse and channels:
        args.append(channels)

    # If vectorization is desired the arguments are overwritten completely.
    if not inverse:
        args = [n_obs, -1]

    return data.reshape(args)

def pad_raster_edges(data, x_pad=0, y_pad=0):
    return np.pad(data, [(0,0),(x_pad,x_pad),(y_pad,y_pad)], constant_values=(0))

def preprocess_raster_resampling(X, y, resample, random_state=None):
    if resample == "upsample":
        print("performing upsample")
        X, y = simple_resample_array(X, y, random_state=random_state)
    elif resample == "downsample":
        print("performing downsample")
        X, y = simple_resample_array(X, y, down=True, random_state=random_state)
    elif resample == "smote":
        print("performing SMOTE")
        sm = SMOTE(sampling_strategy='not majority', random_state=random_state, n_jobs=-1)
        X, y = sm.fit_sample(X, y)
    else:
        print("Ignoring class imbalances")
    return X, y

def generate_keras_model(model, layers=[], compile_kwargs={}):
    model = model
    for layer in layers:
        model.add(layer)
    model.compile(**compile_kwargs)
    model.summary()
    return model

def image_class_accuracy_scores(y_act, y_hat):
    acc = accuracy_score(y_act, y_hat)
    bal_acc = balanced_accuracy_score(y_act, y_hat)
    print(f"Accuracy: {acc}")
    print(f"Balanced Accuracy: {bal_acc}")
    return acc, bal_acc

def image_class_evaluation(model, X, y):
    y_act = np.argmax(y, axis = 1)
    y_hat = np.argmax(model.predict(X), axis=1)
    return image_class_accuracy_scores(y_act, y_hat)