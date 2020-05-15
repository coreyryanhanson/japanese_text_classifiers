def color_values_to_float(array, bits):
    """Converts arrays with integer color data to floats depending on their bit depth. Currently only supports
    8bit, 10 bit, and 12 bit images."""

    bit_dict = {8:255, 10:1023, 12:4095}
    return array/bit_dict[bits]


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

