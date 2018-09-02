import math
import time

from PIL import Image as pil_image
import numpy as np
import cv2


def quantize_lab_image(lab_image, bins, max_value):
    # type: (np.ndarray, int) -> np.ndarray

    ab_channels = lab_image[:, :, 1:]

    division_factor = math.ceil(max_value / bins)

    ab_channels = (ab_channels // division_factor) - 1
    ab_channels[:, :, 0] = ab_channels[:, :, 0] * bins

    indexes = ab_channels[:, :, 0] + ab_channels[:, :, 1]

    image_shape = ab_channels.shape

    final_result = np.zeros((image_shape[0], image_shape[1]) + (bins**2,))

    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            index = int(indexes[i, j])
            pixel_distribution = np.zeros(bins**2)
            pixel_distribution[index] = 1.0
            final_result[i, j] = pixel_distribution

    return final_result


def convert_quantization_to_image(quantization, bins):
    # type: (np.ndarray, int) -> np.ndarray
    image_shape = (quantization.shape[0], quantization.shape[1])

    indexes = np.argmax(quantization, axis=2)
    a_channel = (indexes // bins) + 1
    b_channel = (indexes % bins) + 1

    a_channel = a_channel.reshape(image_shape[0], image_shape[1], 1)
    b_channel = b_channel.reshape(image_shape[0], image_shape[1], 1)

    return np.concatenate((a_channel, b_channel), axis=2)