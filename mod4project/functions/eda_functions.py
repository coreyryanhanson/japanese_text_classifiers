import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from IPython.display import HTML
from io import BytesIO
from base64 import b64encode

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def common_plot_setup(figure=None, axes=None, palette=None, font=None, figsize=(15, 10), style=None):
    if style:
        plt.style.use(style)
    if font:
        plt.rcParams['font.family'] = font
    if palette:
        sns.set_palette(palette)
    if figure and axes:
        fig, ax = figure, axes
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(figsize)
    return fig, ax


def keras_fit_history_to_df(results, epoch_offset=0):
    epochs = pd.Index(results.epoch) + epoch_offset
    return pd.DataFrame(results.history, index=epochs)


def plot_keras_fit_history(results, epoch_offset=0, common_plot_kwargs={}):
    fig, ax = common_plot_setup(**common_plot_kwargs)
    history = pd.DataFrame()
    if type(results) == list:
        for result in results:
            history_slice = keras_fit_history_to_df(result, epoch_offset)
            epoch_offset = 1+history_slice.index[-1]
            history = history.append(history_slice)
    else:
        history = keras_fit_history_to_df(results, epoch_offset)
    sns.lineplot(data=history, ax=ax)
    plt.show()
    return history

def map_char_codes(labels, classmap):
    df = pd.DataFrame(pd.Series(labels, name="index")).copy()
    return df.merge(classmap, how="left", on="index")["char"]

def plot_keras_test_evaluation(listed_scores, index_offset=0, epoch_per_test=1, common_plot_kwargs={}):
    fig, ax = common_plot_setup(**common_plot_kwargs)
    history = pd.DataFrame(listed_scores, columns=["test_loss", "test_accuracy"])
    history.index = history.index * epoch_per_test + index_offset
    sns.lineplot(data=history, ax=ax)
    plt.show()
    return history

def pil_to_html_img_tag(image):
    b = BytesIO()
    image.save(b, format='png')
    return f"<img src='data:image/png;base64,{b64encode(b.getvalue()).decode('utf-8')}'/>"


def bulk_character_viewer(data, labels, indices=(1), predictions=None, columns=3):
    """Allows viewing of multiple images and their labels/predictions in a single cell divided into columns."""
    shape = data.shape

    # Sets aside a simple boolean to prevent index errors if predictions are omitted
    skip_pred = type(predictions) != pd.core.series.Series

    # Checks and adjusts to make sure the color channel is included.
    if shape != 4:
        data = data.reshape([shape[0], shape[1], shape[2], 1]).copy()

    # Loops through the range adding html lines for a raw image and it's labels.
    code_lines = []
    for i in np.arange(*indices):
        img, label = pil_to_html_img_tag(array_to_img(data[i])), f"<p>{labels[i]}</p>"
        if skip_pred:
            line = f"<span>{img} {label}<br></span>"
        else:
            label = f"<p>Actual: {labels[i]}</p>"
            prediction = f"<p>Predicted: {predictions[i]}</p>"
            line = f"<span>{img}<p>{predictions.index[i]}</p> {label} {prediction}<br></span>"
        code_lines.append(line)

    # Puts the list of lines into a block of HTML
    code = "\n".join(code_lines)

    # Returns html seperated by the amount of columns.
    return HTML(f"<div style='column-count: {columns};'>{code}</div>")


def colors_from_values(values, palette_name):
    """Creates a palette that changes coloring of bar plots to vary by their y value"""

    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)