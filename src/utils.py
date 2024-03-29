"""Module providing common utilities used in the other modules."""

from colorsys import hsv_to_rgb


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
