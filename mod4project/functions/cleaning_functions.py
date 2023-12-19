import os
import gzip
import numpy as np

from imblearn.over_sampling import SMOTE
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import array_to_img

from geomdl import BSpline
from geomdl.utilities import generate_knot_vector

from random_lumberjacks.src.random_lumberjacks.model.model_classes import *

def color_values_to_float(array, bits):
    """Converts arrays with integer color data to floats depending on their bit depth. Currently only supports
    8bit, 10 bit, and 12 bit images."""

    bit_dict = {8:255, 10:1023, 12:4095}
    return array/bit_dict[bits]

def create_classification_dirs(path_list, labels):
    """Creates multiple subdirectories for each directory in a list of paths."""

    for path in path_list:
        for label in labels:
            os.makedirs(os.path.join(path, label))

def export_numpy_array_to_images(data, labels, parent_path, prefix=""):
    for label in np.unique(labels):
        new_dir = f"{prefix}{label}"
        new_path = os.path.join(parent_path, new_dir)
        os.mkdir(new_path)
        mask = labels == label
        for i, observation in enumerate(data[mask]):
            img = array_to_img(observation)
            img.save(os.path.join(new_path, f"{i:04}.png"))

def fill_np_array_nan_neighbors(array, needs_transpose=False):
    """Fills a numpy array with nan values with interpolations from neighboring points."""

    # Creates a mask of na values in array.
    mask = np.isnan(array)

    # If nans are detected, it fills any missing values first with interpolation between existing points.
    if np.any(mask):

        # Transposes the arrays for interpolation if needed.
        if needs_transpose:
            array = array.T
            mask = mask.T

        array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])

        # Reverses the transpose if needed.
        if needs_transpose:
            array = array.T
    return array

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


def lstm_angle_components(stroke_list):
    """Iterates through a list of numpy arrays for each stroke calculating the sine and cosine of the angle between each point."""

    stroke_list = stroke_list.copy()
    angle_components = []
    for stroke in stroke_list:
        distance = np.diff(stroke, axis=0)
        hypot = np.hypot(distance[:, 0], distance[:, 1])

        # Calculates the sin and cosine of the angle. Replaces nans with interpolations from neighboring points if zero division error.
        angle_group = fill_np_array_nan_neighbors((distance.T / hypot).T, True)
        angle_group = np.concatenate((angle_group, np.expand_dims(angle_group[-1], axis=0)), axis=0)
        angle_components.append(angle_group)
    return angle_components


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


def preprocess_lstm_pipeline(array, labels, test_size=None, val_size=None, random_seed=None):
    nobs, sequence_length, nfeatures = array.shape

    # App provides possible values range from -1 to 1. The sin and cosine values of the angles also contain this range. This will
    # standardize the data for machine learning, while also preserving information present in the relative size of a drawing.
    scaled_array = (array.copy() + 1) / 2
    labels = labels.copy()

    array1d = scaled_array.reshape([nobs, -1])

    if test_size and val_size:
        print("Performing a train, test, validation split.")
        X_int, X_test, y_int, y_test = train_test_split(array1d, labels, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X_int, y_int, test_size=val_size, random_state=random_seed)
        return X_train.reshape([-1, sequence_length, nfeatures]), X_val.reshape(
            [-1, sequence_length, nfeatures]), X_test.reshape([-1, sequence_length, nfeatures]), to_categorical(
            y_train), to_categorical(y_val), to_categorical(y_test)
    if test_size or val_size:
        print("Performing a train, test split.")
        test_size = max([test_size, val_size])
        X_train, X_test, y_train, y_test = train_test_split(array1d, labels, test_size=test_size,
                                                            random_state=random_seed)
        return X_train.reshape([-1, sequence_length, nfeatures]), X_test.reshape(
            [-1, sequence_length, nfeatures]), to_categorical(y_train), to_categorical(y_test)
    else:
        print("Skipping train, test, split")
        return scaled_array, to_categorical(labels)

        # One hot encodes the labels in order to be fit to the lstm.
    labels = to_categorical(labels.copy())

def render_coordinates_file_to_img(coord_list, resolution=(28, 28), stroke_width=2):
    """takes a list of strokes and uses Pillow's image draw function after scaling them to fill the desired resolution."""

    # Scales linedata to have a centered maximum fit within the desired resoltion.
    scaled = scale_points_for_pixels(coord_list, resolution, stroke_width)

    img = Image.new('L', resolution, color=0)
    draw = ImageDraw.Draw(img)
    for coords in scaled:
        line_from_array(draw, coords, width=stroke_width)
    return img


def resample_coords_smooth(coords, sample_size=120, degree=3):
    """Resamples an array of coordinates while smoothing out the new values with a b-spline."""

    #Prevents code from breaking when the stroke contains too few values.
    while coords.shape[0] < degree + 1:
        coords = np.concatenate((coords, np.expand_dims(coords[-1], axis=0)), axis=0)

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


def strokes_to_array(data, max_strokes=40):
    """Stacks the stroke data across dimensions. It keeps consistency among the different characters by filling in zeros
    when there is insufficient strokes. The amount of alloted features is defined by the parameter "max fetures" which
    can be calculated by taking the total amount of desired strokes multiplied by 4 (x, y, cosΘ, and sinΘ,)"""

    sample_size = data[0].shape[0]
    stroke_count = len(data)
    max_features = max_strokes * 4

    angle_components = lstm_angle_components(data)

    # Groups X and Y values for different features within the same dimension.
    stacked_data = np.dstack(data)
    stacked_angles = np.dstack(angle_components)

    new_array = np.concatenate([stacked_data, stacked_angles], axis=1)

    # Prevents code from breaking if an observation has more strokes than anticipated.
    if stroke_count > max_strokes:
        print(f"{stroke_count} strokes exceed maximum of {max_strokes}. Trimming array")
        new_array = new_array[:, :, :max_strokes]
        stroke_count = max_strokes

    # Pads the array with zeros where there are less strokes than the maximum.
    padded = np.pad(new_array, [(0, 0), (0, 0), (0, max_strokes - stroke_count)], mode='constant', constant_values=0)
    flattened = padded.reshape([sample_size, -1])

    return flattened.reshape([1, sample_size, max_features])