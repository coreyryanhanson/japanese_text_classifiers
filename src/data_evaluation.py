"""A module that contains methods to assist examining the results after
training a model."""

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from .data_loading import StrokeDataset
from .manage_models import CharacterTrainer
from .utils import HueShifter


class AnimatedStrokes:
    """Uses Matplotlib to visualize animated drawing of characters.
    Args:
        strokes (list[list[list[float]]]): A nested list of stroke data.
        hue_shift (int, optional): Strokes will be represented by hue shifted
            colors. Value of 0-360 indicating the amount of degrees to shift.
            Defaults to 40.
        point_size (int, optional): The size of the placed scatterplot points
            that represent captured datapoints. Defaults to 20.
        point_color (tuple[float, ...], optional): Scatterplot point color.
            Defaults to (0.5, 0.5, 0.5, 0.5).
        line_width (int, optional): The line width of the character animated
            image. Defaults to 3.
    """
    def __init__(self,
                 strokes: list[list[list[float]]],
                 hue_shift: int = 40,
                 point_size: int = 20,
                 point_color: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5),
                 line_width: int = 3,
                 force_stop_on_refresh: bool = False
                 ) -> None:
        self.strokes = self._set_strokes(strokes)
        self.hue_shift = hue_shift
        self.point_size = point_size
        self.point_color = point_color
        self.line_width = line_width
        self._timesteps = self._set_timesteps(strokes)
        self._anim: Optional[FuncAnimation] = None

    def _set_strokes(self,
                     strokes: list[list[list[float]]]
                     ) -> list[npt.NDArray[np.float32]]:
        return [np.array(stroke, dtype=np.float32) for stroke in strokes]

    def _set_timesteps(self,
                       strokes: list[list[list[float]]]
                       ) -> npt.NDArray[np.int32]:
        return np.cumsum([0] + [len(stroke)
                                for stroke in strokes], dtype=np.int32)

    def _build_axes(self, figwidth: float) -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(figwidth, figwidth))
        xmin, xmax = -1, 1
        ymin, ymax = -1, 1
        ax = plt.axes((0., 0., 1., 1.), xlim=(xmin, xmax), ylim=(ymin, ymax))
        return fig, ax

    def _create_crosshairs(self,
                           ax: plt.Axes,
                           span: float = 0.9,
                           dashes: tuple[int, int] = (4, 8)
                           ) -> None:
        for x, y in [[[0, 0], [-span, span]], [[-span, span], [0, 0]]]:
            ax.plot(x,
                    y,
                    linestyle="dashed",
                    dashes=dashes,
                    color=self.point_color,
                    linewidth=self.line_width/2)

    def _set_style(self,
                   fig: plt.Figure,
                   ax: plt.Axes
                   ) -> tuple[plt.Figure, plt.Axes]:
        ax.axis('off')
        fig.patch.set_facecolor("#000000")
        # ax.set_facecolor((0, 0, 0))
        self._create_crosshairs(ax)
        return fig, ax

    def _build_lineplots(self, ax: plt.Axes) -> list[Line2D]:
        hue_shift = HueShifter(hue_step=self.hue_shift)
        plots = []
        for _ in range(len(self.strokes)):
            plots.append(ax.plot([],
                                 [],
                                 color=hue_shift.get_rgb(),
                                 linewidth=self.line_width)[0])
            hue_shift.step()
        return plots

    def _build_scatterplots(self, ax: plt.Axes) -> list[PathCollection]:
        return [ax.scatter([], [], s=self.point_size, color=self.point_color)
                for _ in range(len(self.strokes))]

    def _create_animated_stroke(self,
                                fig: plt.Figure,
                                ax: plt.Axes,
                                interval: int = 50,
                                repeat_delay: int = 1000
                                ) -> FuncAnimation:
        def update(frame: int) -> tuple[list[PathCollection], list[Line2D]]:
            if frame == 0:
                for i in range(len(self.strokes)):
                    plots[0][i].set_offsets(self.strokes[i][:0])
                    plots[1][i].set_xdata(self.strokes[i][:0, 0])
                    plots[1][i].set_ydata(self.strokes[i][:0, 1])
                return plots
            i = np.searchsorted(self._timesteps,
                                frame + 1,
                                side="left",
                                sorter=None) - 1
            data = self.strokes[i][:frame-self._timesteps[i]]
            plots[0][i].set_offsets(data)
            plots[1][i].set_xdata(data[:, 0])
            plots[1][i].set_ydata(data[:, 1])
            return plots

        frames = self._timesteps[-1]
        scatterplots = self._build_scatterplots(ax)
        lineplots = self._build_lineplots(ax)
        plots = (scatterplots, lineplots)
        return FuncAnimation(fig,
                             update,
                             repeat_delay=repeat_delay,
                             interval=interval,
                             frames=frames,
                             blit=True,
                             repeat=True)

    def force_stop_animation(self) -> None:
        """Forces the animation object to stop. Useful if multiple instances
        of the object are called and old ones are hindering performance.
        """
        if self._anim is not None:
            self._anim._stop()
            self._anim = None

    def plot(self,
             figwidth: float = 6,
             interval: int = 50,
             repeat_delay: int = 3000
             ) -> None:
        """Plots a vizualization of the stroke data animated using matplotlib.

        Args:
            figwidth (float, optional): The one dimension of the figure size.
                Output will be square. Defaults to 6.
            interval (int, optional): Corresponds to the interval value in the
                matplotlib FuncAnimation object. Defaults to 50.
            repeat_delay (int, optional): Corresponds to the repeat delay value
                in the matplotlib FuncAnimation object. Defaults to 3000.
        """
        fig, ax = self._build_axes(figwidth)
        fig, ax = self._set_style(fig, ax)
        self._anim = self._create_animated_stroke(fig,
                                                  ax,
                                                  interval,
                                                  repeat_delay)
        plt.show()


class IncorrectCharacters:
    """Class that is used to better format the raw data from the trainer's
    "check_predictions" method and to quickly preview an animation of points
    drawn in order to quickly identify mistakes in the training data and
    general shortcomings of the model.

    Args:
        character_trainer (CharacterTrainer): Custom CharacterTrainer class
            that manages the torch training loops.
        dataset (StrokeDataset): Custom StrokeDataset class that has additional
            methods to quickly bypass transforms and find the raw file paths.
        class_dict (pd.DataFrame): A dataframe containing details about each
            integer coded class.
        temperature (float, optional): Used to either enhance or dampen
            differences between the prediction probabilities. Should be a
            number greater than 0 where values < 1 will exaggerate and
            values > 1 will dampen. Defaults to 1.
    """
    def __init__(self,
                 character_trainer: CharacterTrainer,
                 dataset: StrokeDataset,
                 class_dict: pd.DataFrame,
                 temperature: float = 1
                 ) -> None:
        self._misses = character_trainer.check_predictions(dataset,
                                                           temperature)
        self._dataset = dataset
        self._classes = class_dict
        self._anim: Optional[AnimatedStrokes] = None

    def __len__(self) -> int:
        return len(self._misses)

    def _get_label_details(self, i: int) -> pd.DataFrame:
        strokes, label = self._dataset.get_raw(i)
        label_df = pd.DataFrame(self._classes.iloc[label]).T
        label_df.insert(4, "strokes_found", len(strokes))
        return label_df

    def compare(self, i: int, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Picks one of the incorrect predictions showing the full detail about
            the actual label and n guesses with next highest probabilities.

        Args:
            i (int): Index of incorrect prediction to examine.
            top_n (int): Determines the n next highest values to show after
                the softmax has been applied.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Information about the actual
            label and a table of softmax results.
        """
        actual, probabilities = self._misses[i]
        actual = self._get_label_details(i)
        probabilities, indices = torch.topk(probabilities,
                                            top_n,
                                            dim=1,
                                            largest=True,
                                            sorted=True)
        probabilities = pd.DataFrame(probabilities.t(),
                                     index=indices.squeeze().numpy(),
                                     columns=["probability"])
        probabilities = probabilities.join(self._classes, how="left")
        return actual, probabilities

    def examine(self, i: int,
                figwidth: float = 6,
                force_stop_previous: bool = False
                ) -> str:
        """Function to provide useful metrics when evaluating an incorrect
        prediction. Shows the full label information, the path of the file,
        and renders an animation of the points and lines forming the character
        in order.

        Args:
            i (int): Index of incorrect prediction to examine.
            figwidth (float, optional): The one dimension of the figure size.
                Output will be square. Defaults to 6.
            force_stop_previous (bool, optional): If the matplotlib
                FuncAnimator object has lingering references, subsequent runs
                will negatively affect future performance. Enable this to
                ensure each time the plot method is run, the old process is
                ended. Defaults to False.

        Returns:
            str: The file path of the data being examined.
        """
        info = self._get_label_details(i)
        path = self._dataset.get_path(i)
        data = self._dataset.get_raw(i)[0]
        if force_stop_previous and self._anim is not None:
            self._anim.force_stop_animation()
        self._anim = AnimatedStrokes(data,
                                     force_stop_on_refresh=force_stop_previous)
        self._anim.plot(figwidth, repeat_delay=1000)
        print(info)
        return path
