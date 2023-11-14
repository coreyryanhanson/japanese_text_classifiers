"""A module to manage the PyTorch data transforms for this project."""

from colorsys import hsv_to_rgb
from typing import Union

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
from scipy.interpolate import make_interp_spline
import torch
import torch.nn.functional as f


class ToBSplines:
    """Transform callable class that will conform a nested list of stroke
    points to a uniform length by smoothing with a B-Spline interpolation.

    Args:
        sample_size (int, optional): The count of sequential points to conform
            each stroke to. Defaults to 120.
        degree (int, optional): The degree of the polynomial that forms the
            piecewise B-spline curve. Defaults to 3.
        smooth_points (float, optional): This sets the strength of the function
            used to evenly distribute the resulting B-spline points. If set
            too high, endpoints may have less fidelity. Defaults to .5.
    """
    def __init__(self,
                 sample_size: int = 120,
                 degree: int = 3,
                 smooth_points: float = .5
                 ) -> None:
        self.sample_size = sample_size
        self.degree = degree
        self.smooth = smooth_points

    def _generate_matched_x_linspace(self,
                                     array: npt.NDArray[np.float32]
                                     ) -> npt.NDArray[np.float32]:
        """By matching the distances on the input dimension to the distances
        between the x and y points, the b-spline interpolation will be a
        smoother curve."""
        array = np.diff(array, axis=0)
        array = np.hypot(array[:, 0], array[:, 1])
        array = np.cumsum(np.concatenate([[0], array]))
        # If the input scale perfectly matches the distance between the x and
        # y points, sometimes the endpoints exhibit deformations. This will
        # mitigate that according to the inverse of the class' smooth_points
        # value.
        dampen = np.linspace(0, array[-1], array.shape[0], dtype=np.float32)
        return array * self.smooth + dampen * (1 - self.smooth)

    def _smooth_resample(self,
                         coords: list[list[float]]
                         ) -> npt.NDArray[np.float32]:
        # Prevents code from breaking when the stroke contains too few values.
        if len(coords) <= self.degree:
            pad = [coords[-1]] * (1 + self.degree - len(coords))
            coords.extend(pad)

        array = np.array(coords, dtype=np.float32)
        x = self._generate_matched_x_linspace(array)
        spl = make_interp_spline(x, array, self.degree)
        return spl(np.linspace(0, x[-1], self.sample_size, dtype=np.float32))

    def __call__(self,
                 sample: list[list[list[float]]]
                 ) -> npt.NDArray[np.float32]:
        output = [self._smooth_resample(value) for value in sample]
        return np.stack(output, axis=0)


class ArrayToTensor:
    """Simple Transform class to convert a numpy array to PyTorch Tensor.
    """
    def __call__(self, sample: list[float]) -> torch.Tensor:
        return torch.FloatTensor(sample)


class ExtractAngles:
    """Transform class that takes a tensor of shape
    [stroke, sequence, 2(x and y)] and extracts angles with an output shape
    of either [stroke, sequence -1, new outputs (angles + optional magnitude)]
    or [stroke, sequence, orig and new outputs (angles + optional magnitude)]

    Args:
        include_magnitudes (bool, optional): Whether or not data for the
            distance between points is included. Defaults to False.
        keep_original (bool, optional): Whether or not to replace the original
            data (and take one item out of the sequence) or append results to
            the original data (repeating the last sequence item on the new
            data). Defaults to True.
    """
    def __init__(self,
                 include_magnitudes: bool = False,
                 keep_original: bool = True
                 ) -> None:
        self._include_magnitudes = include_magnitudes
        self._keep_original = keep_original

    def _calc_angular_difference(self, tensor: torch.Tensor) -> torch.Tensor:
        # The difference between the subsequent value and the value.
        tensor = torch.diff(tensor, n=1, dim=1)
        # The length between each point.
        magnitude = torch.hypot(tensor[:, :, 0], tensor[:, :, 1]).unsqueeze(-1)
        # The (absolute) angle from point to subsequent point expressed
        # in the form of [cosine, sine].
        tensor = tensor.div(magnitude)
        if self._include_magnitudes:
            tensor = torch.cat([tensor, magnitude], dim=-1)
        return tensor

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        angles = self._calc_angular_difference(sample)
        if self._keep_original:
            # The difference function clips a value. In order to match original
            # data. The last item in the sequence is repeated once.
            angles = torch.cat([angles, angles[:, -1].unsqueeze(1)], dim=1)
            angles = torch.cat([sample, angles], dim=-1)
        return angles


class EmptyStrokePadder:
    """Torch transform that conforms a single dimension by padding specified
    values to the end of a single axis.

    Args:
        stroke_count (int, optional): The size of the axis after padding is
            complete. Defaults to 40.
        pad_dim (int, optional): The dimension to pad. Defaults to 0.
        pad_value (float, optional): The constant values of the pad. Defaults
            to 0.
    """
    def __init__(self,
                 stroke_count: int = 40,
                 pad_dim: int = 0,
                 pad_value: float = 0
                 ) -> None:
        self.stroke_count = stroke_count
        self.pad_dim = pad_dim
        self.pad_value = pad_value

    def _specify_pad(self, sample: torch.Tensor) -> list[int]:
        pad_size = self.stroke_count - sample.shape[0]
        pad = [0] * 2 * len(sample.shape)
        pad_dim = -self.pad_dim * 2 - 1
        pad[pad_dim] = pad_size
        return pad

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        n_strokes = sample.shape[0]
        if n_strokes == self.stroke_count:
            return sample
        if n_strokes > self.stroke_count:
            raise RuntimeError("stroke_count cannot be less than the"
                               "actual number of strokes")
        pad = self._specify_pad(sample)
        return f.pad(sample, pad, mode="constant", value=self.pad_value)


class StrokesToPil:
    """Rasterizes raw stroke data to a PIL image.
    Args:
        output_resolution (tuple[int, int]): The resolution for the output
            image.
        stroke_width (int, optional): The stroke width in pixels. Defaults
            to 2.
        invert_y (bool, optional): Whether or not to perform a verticle flip
            if the coordinate systems are reversed. Defaults to True.
        multicolor (bool, optional): Whether output pixels with a different hue
            for each stroke or to keep as a grayscale image. Defaults to False.
        mc_hue_shift (int, optional): When multicolor is enabled, this
            determines the amount to shift the hue after each stroke. Defaults
            to 10.
    """
    def __init__(self,
                 output_resolution: tuple[int, int],
                 stroke_width: int = 2,
                 invert_y: bool = True,
                 multicolor: bool = False,
                 mc_hue_shift: int = 10,
                 ) -> None:
        self._resolution = output_resolution
        self._stroke_width = stroke_width
        self._invert_y = invert_y
        self._multicolor = multicolor
        self._hue_shift = mc_hue_shift

    def _draw_line(self,
                   draw: ImageDraw.ImageDraw,
                   points: npt.NDArray[np.float32],
                   fill: Union[tuple[int, ...], int]
                   ) -> None:
        coords = points.reshape(-1).tolist()
        draw.line(coords, fill, self._stroke_width)

    def _create_image(self,
                      points: Union[list[npt.NDArray[np.float32]],
                                    npt.NDArray[np.float32]]
                      ) -> Image.Image:
        mode = "RGB" if self._multicolor else "L"
        img = Image.new(mode, self._resolution, color=0)
        draw = ImageDraw.Draw(img)
        if self._multicolor:
            hue = 0
            for stroke in points:
                fill = tuple(round(channel*255)
                             for channel in hsv_to_rgb(hue/360, 1, 1))
                self._draw_line(draw, stroke, fill=fill)
                hue += self._hue_shift
        else:
            for stroke in points:
                self._draw_line(draw, stroke, fill=255)
        return img

    def _extract_bounds(self,
                        points: Union[list[npt.NDArray[np.float32]],
                                      npt.NDArray[np.float32]]
                        ) -> tuple[tuple[float, float], tuple[float, float]]:
        points = np.vstack(points).T
        x_min, x_max = points[0].min(), points[0].max()
        y_min, y_max = points[1].min(), points[1].max()
        x_span, y_span = x_max - x_min, y_max - y_min
        x_cent, y_cent = (x_min + x_max) / 2, (y_min + y_max) / 2
        # To facilitate a better fit.
        if x_span > y_span:
            return (x_min, x_max), (y_cent - x_span / 2, y_cent + x_span / 2)
        else:
            return (x_cent - y_span / 2, x_cent + y_span / 2), (y_min, y_max)

    def _rescale_points(self,
                        points: Union[list[npt.NDArray[np.float32]],
                                      npt.NDArray[np.float32]]
                        ) -> list[npt.NDArray[np.float32]]:
        """Scales linedata to have a centered maximum fit within the desired
        resolution."""
        x_bounds, y_bounds = self._extract_bounds(points)
        # To prevent clipping of edges, the stroke width must be accounted for.
        buffer = self._stroke_width
        # Loops through setting dimensions to output size.
        new_points = []
        for stroke in points:
            stroke = np.array(stroke)
            scaled_x = np.interp(stroke.T[0],
                                 x_bounds,
                                 (0 + buffer, self._resolution[0] - buffer))
            scaled_y = np.interp(stroke.T[1],
                                 y_bounds,
                                 (0 + buffer, self._resolution[1] - buffer))
            new_points.append(np.stack((scaled_x,
                                        scaled_y)).T.astype(np.float32))
        return new_points

    def __call__(self,
                 sample: Union[list[npt.NDArray[np.float32]],
                               npt.NDArray[np.float32]]
                 ) -> Image.Image:
        scaled = self._rescale_points(sample)
        img = self._create_image(scaled)
        if self._invert_y:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return img
