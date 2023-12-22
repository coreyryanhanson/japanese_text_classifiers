"""Module providing common utilities used in the other modules."""

from colorsys import hsv_to_rgb

import torch
import torch.nn.functional as f


class HueShifter:
    """Custom class to manage automatic hue shifted colors.

    Args:
        initial_hue (int, optional): The initial hue (in degrees). Defaults
            to 0.
        hue_step (int, optional): Amount to increment hue (in degrees).
            Defaults to 10.
        saturation (float, optional): Saturation level (from 0 to 1). Defaults
            to 1.
        value (float, optional): Brightness level (from 0 to 1). Defaults to 1.
    """
    def __init__(self,
                 initial_hue: int = 0,
                 hue_step: int = 10,
                 saturation: float = 1,
                 value: float = 1
                 ) -> None:
        self._hue_degrees = initial_hue
        self._saturation = saturation
        self._value = value
        self._hue_step = hue_step

    def step(self) -> None:
        """Increments the stored hue value by the stored shift amount."""
        self._hue_degrees += self._hue_step

    def get_hsv(self) -> tuple[float, float, float]:
        """Gets an HSV tuple of values ranging from 0 to 1.

        Returns:
            tuple[float, float, float]: HSV color channels as floats.
        """
        return self._hue_degrees / 360, self._saturation, self._value

    def get_rgb(self) -> tuple[float, ...]:
        """Gets an RGB tuple of values ranging from 0 to 1.

        Returns:
            tuple[float, ...]: RGB color channels as floats.
        """
        return hsv_to_rgb(*self.get_hsv())

    def get_rgb_ints(self) -> tuple[int, ...]:
        """Gets an RGB tuple of values ranging from 0 to 255.

        Returns:
            tuple[int, ...]: RGB color channels as 8 bit ints.
        """
        return tuple(round(channel * 255)
                     for channel in hsv_to_rgb(*self.get_hsv()))


def pad_constant_to_max_right(tensor: torch.Tensor,
                              max_pad: int,
                              dim: int,
                              value: float = 0
                              ) -> torch.Tensor:
    """Conforms a single dimension to a specified max size by padding indicated
    values to the end of a single axis.

    Args:
        tensor (torch.Tensor): The tensor to pad.
        max_pad (int): The size to modify the specified dimension to with
            padding. Should exceed the current size.
        dim (int): The dimension to pad.
        value (float, optional): The constant value for padding. Defaults to 0.

    Returns:
        torch.Tensor: The tensor with specified values padded at the end of the
        specified dimension to a set size.
    """
    pad = _get_padright_to_max_args(tensor, max_pad, dim)
    return f.pad(tensor, pad, mode="constant", value=value)


def _get_padright_to_max_args(tensor: torch.Tensor,
                              max_pad: int,
                              dim: int
                              ) -> list[int]:
    pad_size = max_pad - tensor.shape[dim]
    pad = [0] * 2 * len(tensor.shape)
    pad_dim = -dim * 2 - 1
    pad[pad_dim] = pad_size
    return pad
