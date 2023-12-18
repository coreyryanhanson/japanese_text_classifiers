"""A module to manage the PyTorch data transforms for this project."""

from colorsys import hsv_to_rgb
from typing import Union

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
from scipy.interpolate import make_interp_spline
import torch
import torch.nn.functional as f

from .utils import HueShifter


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
        pad_size = self.stroke_count - sample.shape[self.pad_dim]
        pad = [0] * 2 * len(sample.shape)
        pad_dim = -self.pad_dim * 2 - 1
        pad[pad_dim] = pad_size
        return pad

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        n_strokes = sample.shape[self.pad_dim]
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
            hues = HueShifter(initial_hue=0, hue_step=self._hue_shift)
            for stroke in points:
                fill = hues.get_rgb_ints()
                self._draw_line(draw, stroke, fill=fill)
                hues.step()
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


class _SwitchedInputTransform:
    def __init__(self, input_idx) -> None:
        self._input_idx = input_idx

    def _main_func(self, sample: torch.Tensor) -> torch.Tensor:
        return sample

    def __call__(self,
                 samples: tuple[torch.Tensor, ...]
                 ) -> tuple[torch.Tensor, ...]:
        as_list = list(samples)
        as_list[self._input_idx] = self._main_func(as_list[self._input_idx])
        return tuple(as_list)


class InputNormalizer(_SwitchedInputTransform):
    """Normalizes a dimension of a single tensor among multiple.
    Args:
        input_idx (int): The index of the input iterable indicating where the
            tensor is located.
        dim (int): The dimension to normalize.
    """
    def __init__(self, input_idx: int, dim: int) -> None:
        super().__init__(input_idx)
        self._dim = dim

    def _main_func(self, sample: torch.Tensor) -> torch.Tensor:
        return f.normalize(sample, dim=self._dim)


class InputMinMaxTransformer(_SwitchedInputTransform):
    """Normalizes a dimension of a single tensor among multiple.
    Args:
        input_idx (int): The index of the input iterable indicating where the
            tensor is located.
        old_min (torch.Tensor): Values of the dataset minimum to adjust data
            elementwise.
        old_max (torch.Tensor): Values of the dataset maximum to adjust data
            elementwise.
        new_min (float, optional): Scalar of what to adjust the new minimum to.
            Defaults to 0.
        new_max (float, optional): Scalar of what to adjust the new maximum to.
            Defaults to 1.
    """
    def __init__(self,
                 input_idx: int,
                 old_min: torch.Tensor,
                 old_max: torch.Tensor,
                 new_min: float = 0,
                 new_max: float = 1
                 ) -> None:
        super().__init__(input_idx)
        self._old_min = old_min
        self._new_min = new_min
        old_span = old_max - old_min
        new_span = new_max - new_min
        self._scale = new_span / old_span

    def _main_func(self, sample: torch.Tensor) -> torch.Tensor:
        return (sample - self._old_min) * self._scale + self._new_min


class InputRecenter(_SwitchedInputTransform):
    """Simple elementwise recenter by subtracting the input mean/skew.

    Args:
        input_idx (int): The index of the input iterable indicating where the
            tensor is located.
        skew (torch.FloatTensor): Values of the dataset mean to be subtracted
            elementwise.
    """
    def __init__(self, input_idx: int, skew: torch.FloatTensor) -> None:
        super().__init__(input_idx)
        self._skew = skew

    def _main_func(self, sample: torch.Tensor) -> torch.Tensor:
        return sample - self._skew


class InputGaussianNoise(_SwitchedInputTransform):
    """Adds Gaussian Noise to a specific input scaled to the indicated standard
    deviation.

    Args:
        input_idx (int): The index of the input iterable indicating where the
            tensor is located.
        std (Union[float, torch.Tensor]): The standard deviation of the
            gaussian noise to add.
    """
    def __init__(self,
                 input_idx: int,
                 std: Union[float, torch.Tensor]
                 ) -> None:
        super().__init__(input_idx)
        self._std = std

    def _generate_noise(self, shape: torch.Size) -> torch.Tensor:
        return torch.randn(shape) * self._std

    def _main_func(self, sample: torch.Tensor) -> torch.Tensor:
        return sample + self._generate_noise(sample.shape)


class StrokeExtractAbsolute:
    """A transform class that divides the (stroke, point, features) structure
    of the dataset to a pair of relative and absolute features. The relative
    features are intended to be generalizable between strokes and will keep
    the same shape until the model reshapes them to mix with the batches. The
    absolute features are aggregations among the points in order to be fed
    as additional parameters directly into a stroke based GRU that will be
    concatenated with the final outputs of the lstm's hidden layer and have
    a simplified shape of (stroke, features).
    """
    def __init__(self) -> None:
        self._angles = ExtractAngles(include_magnitudes=True,
                                     keep_original=False)

    def _extract_mean_std(self,
                          tensor: torch.Tensor,
                          agg_dim: int
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        # By extracting the mean and standard deviation into single pairs of
        # x and y values, the initial lstm can reduce some of its complexity
        # and generalize to the basic shapes instead of retaining information
        # about their absolute position and size.
        means = torch.mean(tensor, dim=agg_dim)
        stds = torch.std(tensor, dim=agg_dim)
        return means, stds

    def _extract_angle_mean(self,
                            tensor: torch.Tensor,
                            agg_dim: int
                            ) -> torch.Tensor:
        # Likewise the angles can have their overall directionallity extracted
        # into a single pair of sine and cosine averages where the angles
        # between pairs of points are weighted by the distance between them to
        # best represent the information in the drawn shape.
        angles = tensor[:, :, :2]
        weights = tensor[:, :, 2]
        summed = (angles * weights.unsqueeze(-1)).sum(dim=agg_dim)
        return summed / weights.sum(dim=agg_dim).unsqueeze(-1)

    def _sin_to_relative(self, sin_values: torch.Tensor) -> torch.Tensor:
        # Angles will be normalized to values between -1 (-90°) and 1 (90°).
        # Since they will ultimately measure the relative angle from a point
        # to an existing pair of points, high values will be very uncommon and
        # can safely be represented by a single value.
        angles = torch.asin(sin_values) * (2 / np.pi)
        # Calculates the relative difference between absolute angles
        angles = torch.diff(angles, n=1, dim=-1)
        # Function repeats and there should be no values outside of (-1 to 1).
        # Edge cases from difference operation can be offset.
        angles = torch.where(angles > 1, angles - 2, angles)
        angles = torch.where(angles < -1, angles + 2, angles)
        return angles

    def _extract_relative_angles(self, tensor: torch.Tensor) -> torch.Tensor:
        relative_angles = self._sin_to_relative(tensor[:, :, 1])
        # The difference operation eliminates a value so the first value will
        # always be assumed to be zero.
        zeros = torch.zeros((relative_angles.shape[0], 1), dtype=torch.float32)
        return torch.cat([zeros, relative_angles], dim=1)

    def _center_mean(self,
                     tensor: torch.Tensor,
                     means: torch.Tensor,
                     agg_dim: int
                     ) -> torch.Tensor:
        return tensor - means.unsqueeze(agg_dim)

    def _convert_relative_angles(self,
                                 tensor: torch.Tensor,
                                 means: torch.Tensor,
                                 agg_dim: int
                                 ) -> torch.Tensor:
        lengths = tensor[:, :, 2:]
        # The relative data includes both the original angles with a simple
        # centering on the averages which informs the direction of a single
        # angle's deviation from its overall average, but it also captures a
        # single value representing distances between angles to better
        # capture smoothness and sharpness of directional changes. Finally,
        # the distance between points is also included to help inform
        # importance any given angle contributes to the overall shape.
        angles = self._center_mean(tensor[:, :, :2], means, agg_dim)
        relative_angles = self._extract_relative_angles(tensor)
        return torch.cat([angles,
                          relative_angles.unsqueeze(-1),
                          lengths], dim=-1)

    def _convert_angles(self,
                        tensor: torch.Tensor,
                        agg_dim: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = self._angles(tensor)
        means = self._extract_angle_mean(angles, agg_dim)
        relative = self._convert_relative_angles(angles, means, agg_dim)
        return relative, means

    def __call__(self,
                 sample: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        agg_dim = 1
        means, stds = self._extract_mean_std(sample, agg_dim)
        relative, angles = self._convert_angles(sample, agg_dim)
        absolute = torch.cat([means, stds, angles], dim=-1)
        return relative, absolute
