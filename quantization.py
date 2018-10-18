import math

import numpy as np


def quantize_lab_image(lab_image, bins, max_value):
    # type: (np.ndarray, int, int) -> np.ndarray

    ab_channels = lab_image[:, :, 1:]

    division_factor = math.ceil(float(max_value) / float(bins))
    ab_channels = (np.floor_divide(ab_channels, division_factor))

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


def convert_quantization_to_image(quantization, bins, max_value):
    # type: (np.ndarray, int) -> np.ndarray
    image_shape = (quantization.shape[0], quantization.shape[1])

    # print(np.sort(np.unique(quantization)))
    sorted_index = np.argsort(-quantization, axis=2)
    indexes = sorted_index[:, :, 0]
    # indexes = np.argmax(quantization, axis=2)
    # indexes = np.zeros((quantization.shape[0], quantization.shape[1]))
    # for i in range(quantization.shape[0]):
    #     for j in range(quantization.shape[1]):
    #         pixel_distribution = quantization[i, j]
    #         nonzero_indexes = np.nonzero(pixel_distribution)
    #         mean = np.mean(nonzero_indexes)
    #         indexes[i, j] = mean

    division_factor = math.floor(float(max_value) / float(bins))
    a_channel = np.floor_divide(indexes, bins)
    b_channel = np.remainder(indexes, bins)

    a_channel = a_channel * float(division_factor)
    b_channel = b_channel * float(division_factor)

    a_channel = a_channel.reshape(image_shape[0], image_shape[1], 1)
    b_channel = b_channel.reshape(image_shape[0], image_shape[1], 1)

    return np.concatenate((a_channel, b_channel), axis=2)
