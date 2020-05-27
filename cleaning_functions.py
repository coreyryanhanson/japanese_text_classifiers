import os
import gzip
import numpy as np

from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import array_to_img

from geomdl import BSpline
from geomdl.utilities import generate_knot_vector

from random_lumberjacks.src.random_lumberjacks.model.model_classes import *

def color_values_to_float(array, bits):
    """Converts arrays with integer color data to floats depending on their bit depth. Currently only supports
    8bit, 10 bit, and 12 bit images."""

    bit_dict = {8:255, 10:1023, 12:4095}
    return array/bit_dict[bits]

def export_numpy_array_to_images(data, labels, parent_path, prefix=""):
    for label in np.unique(labels):
        new_dir = f"{prefix}{label}"
        new_path = os.path.join(parent_path, new_dir)
        os.mkdir(new_path)
        mask = labels == label
        for i, observation in enumerate(data[mask]):
            img = array_to_img(observation)
            img.save(os.path.join(new_path, f"{i:04}.png"))

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

def image_path_extractor(dir_list, parent_path, random_seed=None):
    # Reproducible results
    if random_seed:
        np.random.seed(random_seed)

    # Accounts for single directory searches.
    if type(dir_list) == str:
        dir_list = [dir_list]

    path_list = []
    for directory in dir_list:
        paths = [os.path.join(directory, file) for file in os.listdir(os.path.join(parent_path, directory))]
        path_list.extend(paths)
    np.random.shuffle(path_list)
    return path_list

def image_path_copier(image_path_list, old_parent, new_parent, new_dir_name):
    os.makedirs(os.path.join(new_parent, new_dir_name))
    for img in image_path_list:
        origin = os.path.join(old_parent, img)
        destination = os.path.join(new_parent, new_dir_name, os.path.basename(img))
        shutil.copyfile(origin, destination)
    print(f"{len(image_path_list)} files copied to {os.path.join(new_parent, new_dir_name)}")


def image_path_list_train_test_split(path_list, ratio_train, ratio_test=None):
    """A simple non random train test split for lists of image paths"""

    full_size = len(path_list)
    train_idx = int((full_size) * ratio_train)
    train = path_list[:train_idx]
    if not ratio_test or ratio_test == 1 - ratio_train:
        test = path_list[train_idx:]
        return train, test
    else:
        test_idx = int(full_size - (full_size * ratio_test))
        val = path_list[train_idx:test_idx]
        test = path_list[test_idx:]
        return train, val, test


def parse_emnist(path, offset=16, isimg=True):
    with gzip.open(path) as f:
        # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
        data = np.frombuffer(f.read(), 'B', offset=offset)
    if isimg:
        data = data.reshape(-1, 28, 28)
        return np.swapaxes(data, 1,2)
    else:
        return data


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



def line_from_array(draw, points, fill=255, width=2):
    """Takes a PIL image draw object draws and converts points from a numpy array to render a curved line to those pixels"""

    coords = points.reshape(-1).tolist()
    return draw.line(coords, fill, width)


def parse_to_points_list(points_dict, sample_size=120, degree=3):
    """Converts a dictionary of recorded points to a standard length and vertically reflects them to match PIL's coordinate system."""

    return [resample_coords_smooth([1, -1] * value, sample_size, degree) for key, value in sorted(points_dict.items())]


def resample_coords_smooth(coords, sample_size=120, degree=3):
    """Resamples an array of coordinates while smoothing out the new values with a b-spline."""

    curve = BSpline.Curve()
    curve.degree = degree
    curve.ctrlpts = coords.tolist()
    curve.knotvector = generate_knot_vector(degree, len(curve.ctrlpts))
    curve.sample_size = sample_size
    return np.array(curve.evalpts)


def scale_points_for_pixels(points_list, size=(28, 28), buffer=0):
    """Takes a list of points and normalizes them to pixel size. Optional parameter for buffer around the edges to prevent unintentional cropping (should match stroke width)."""

    all_points = np.vstack(points_list).T
    x_min, x_max = all_points[0].min(), all_points[0].max()
    y_min, y_max = all_points[1].min(), all_points[1].max()
    x_span, y_span = x_max - x_min, y_max - y_min
    x_cent, y_cent = (x_min + x_max) / 2, (y_min + y_max) / 2
    if x_span > y_span:
        y_min, y_max = y_cent - x_span / 2, y_cent + x_span / 2
    else:
        x_min, x_max = x_cent - y_span / 2, x_cent + y_span / 2
    new_points = []
    for points in points_list:
        scaled_x = np.interp(points.T[0], (x_min, x_max), (0 + buffer, size[0] - buffer))
        scaled_y = np.interp(points.T[1], (y_min, y_max), (0 + buffer, size[1] - buffer))
        new_points.append(np.stack((scaled_x, scaled_y)).T)
    return new_points


def strokes_to_array(data, max_features=80):
    """Stacks the stroke data across dimensions. It keeps consistency among the different characters by filling in zeros
    when there is insufficient strokes. The amount of alloted features is defined by the parameter "max fetures" which
    can be calculated by taking the total amount of desired strokes divided by 2 (x and y)"""

    sample_size = data[0].shape[0]
    feature_count = len(data) * 2

    #Groups X and Y values for different features within the same dimension.
    new_array = np.dstack(data).reshape([sample_size, -1])

    #App provides possible values range from -1 to 1. This will standardize the data for machine learning, while also preserving information present in the relative size of a drawing.
    scaled_array = (new_array + 1)/2

    #Prevents code from breaking if an observation has more strokes than anticipated.
    if feature_count > max_features:
        print(f"{feature_count} features exceed maximum of {max_features}. Trimming array")
        scaled_array = scaled_array[:, :max_features]
        feature_count = max_features

    #Pads the array with zeros where there are less features than the maximum.
    padded =  np.pad(scaled_array, [(0, 0), (0, max_features - feature_count)], mode='constant', constant_values=0)
    return padded.reshape([1, sample_size, max_features])