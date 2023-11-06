"""A module to manage the PyTorch data transforms for this project."""

import numpy as np
import numpy.typing as npt
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
    """
    def __init__(self, sample_size: int = 120, degree: int = 3) -> None:
        self.sample_size = sample_size
        self.degree = degree

    # def _smooth_resample_orig(self, coords: list[float]) -> list[float]:
    #    """Slower function will be deprecated in favor of the implementation
    #    from scipy."""
    #    # Prevents code from breaking when the stroke contains too few values.
    #    if len(coords) <= self.degree:
    #        pad = [coords[-1]] * (1 + self.degree - len(coords))
    #        coords.extend(pad)

    #    curve = BSpline.Curve()
    #    curve.degree = self.degree
    #    curve.ctrlpts = coords
    #    curve.knotvector = generate_knot_vector(self.degree,
    #                                            len(curve.ctrlpts))
    #    curve.sample_size = self.sample_size
    #    return curve.evalpts

    def _generate_matched_x_linspace(self,
                                     array: npt.NDArray[np.float32]
                                     ) -> npt.NDArray[np.float32]:
        array = np.diff(array, axis=0)
        array = np.hypot(array[:, 0], array[:, 1])
        array[array == 0] = np.nextafter(np.float16(0.), np.float16(1.))
        return np.cumsum(np.concatenate([[0], array]))

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
    """Torch transform that takes input of [strokes, points, features] and pads
    the output
    """
    def __init__(self, stroke_count: int = 40) -> None:
        self.stroke_count = stroke_count

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        n_strokes = sample.shape[0]
        if n_strokes == self.stroke_count:
            return sample
        if n_strokes > self.stroke_count:
            raise RuntimeError("stroke_count cannot be less than the"
                               "actual number of strokes")
        pad_size = self.stroke_count - n_strokes
        pad = (0, 0, 0, 0, 0, pad_size)
        return f.pad(sample, pad, mode="constant", value=0)
