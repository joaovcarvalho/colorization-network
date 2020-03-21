import math

import numpy as np


def quantize_lab_image(lab_image, bins, max_value=None):
    # type: (np.ndarray, int, int) -> np.ndarray

    if max_value is None:
        max_value = 256

    ab_channels = lab_image[:, :, 1:]

    division_factor = math.floor(float(max_value) / float(bins))
    ab_channels = np.floor_divide(ab_channels, division_factor)

    indexes = ab_channels[:, :, 0] * bins + ab_channels[:, :, 1]

    image_shape = ab_channels.shape

    final_result = np.zeros((image_shape[0], image_shape[1]) + (bins**2,))

    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            index = int(indexes[i, j])
            final_result[i, j, index] = 1.0

    return final_result


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def convert_quantization_to_image(quantization, bins, max_value=None):
    # type: (np.ndarray, int) -> np.ndarray
    if max_value is None:
        max_value = 256

    image_shape = (quantization.shape[0], quantization.shape[1])

    sorted_index = np.argsort(-quantization, axis=2)
    indexes = sorted_index[:, :, 0]

    division_factor = math.floor(float(max_value) / float(bins))
    a_channel = np.floor_divide(indexes, bins)
    b_channel = np.remainder(indexes, bins)

    a_channel = a_channel * float(division_factor)
    b_channel = b_channel * float(division_factor)

    a_channel = a_channel.reshape(image_shape[0], image_shape[1], 1)
    b_channel = b_channel.reshape(image_shape[0], image_shape[1], 1)

    return np.concatenate((a_channel, b_channel), axis=2)


def get_prob(quantization, indices):
    prob = np.zeros(indices.shape + (1,))

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            index = indices[i, j]
            prob[i, j, 0] = quantization[i, j, index]

    return prob


def convert_quantization_to_image_average(quantization, bins, max_value=None, how_many_to_average=1):
    # type: (np.ndarray, int) -> np.ndarray
    image_shape = (quantization.shape[0], quantization.shape[1])

    if max_value is None:
        max_value = 256

    sorted_index = np.argsort(-quantization, axis=2)
    final_image = None
    for i in range(how_many_to_average):
        indexes = sorted_index[:, :, i]
        prob = get_prob(quantization, indexes)

        if np.all(prob == 0):
            return final_image

        division_factor = math.floor(float(max_value) / float(bins))
        a_channel = np.floor_divide(indexes, bins)
        b_channel = np.remainder(indexes, bins)

        a_channel = a_channel * float(division_factor)
        b_channel = b_channel * float(division_factor)

        a_channel = a_channel.reshape(image_shape[0], image_shape[1], 1)
        b_channel = b_channel.reshape(image_shape[0], image_shape[1], 1)

        image = np.concatenate((a_channel, b_channel), axis=2)

        if i == 0:
            final_image = image * prob
        else:
            final_image = final_image + image * prob
    return final_image


def convert_quantization_to_image_expected(quantization, bins, max_value=None):
    image_shape = (quantization.shape[0], quantization.shape[1])

    if max_value is None:
        max_value = 256

    indices = np.ones((image_shape[0] * image_shape[1], 256)) * np.arange(256)
    reshaped_image = quantization.reshape((image_shape[0] * image_shape[1], 256))

    average = np.average(indices, axis=1, weights=reshaped_image)
    indexes = average.reshape(image_shape)

    division_factor = math.floor(float(max_value) / float(bins))
    a_channel = np.floor_divide(indexes, bins)
    b_channel = np.remainder(indexes, bins)

    a_channel = a_channel * float(division_factor)
    b_channel = b_channel * float(division_factor)

    a_channel = a_channel.reshape(image_shape[0], image_shape[1], 1)
    b_channel = b_channel.reshape(image_shape[0], image_shape[1], 1)

    image = np.concatenate((a_channel, b_channel), axis=2)
    return image

