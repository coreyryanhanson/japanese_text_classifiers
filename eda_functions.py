import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



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


def plot_keras_test_evaluation(listed_scores, index_offset=0, epoch_per_test=1, common_plot_kwargs={}):
    fig, ax = common_plot_setup(**common_plot_kwargs)
    history = pd.DataFrame(listed_scores, columns=["test_loss", "test_accuracy"])
    history.index = history.index * epoch_per_test + index_offset
    sns.lineplot(data=history, ax=ax)
    plt.show()
    return history
