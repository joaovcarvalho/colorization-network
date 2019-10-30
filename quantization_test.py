import unittest

import numpy as np

from quantization import convert_quantization_to_image, quantize_lab_image


class TestQuantization(unittest.TestCase):

    def test_small_array(self):
        image = np.ones((10, 10, 3)) * 10.

        quantum = quantize_lab_image(image, bins=256, max_value=256)
        final_image = convert_quantization_to_image(quantum, bins=256, max_value=256)

        l_channel = image[:, :, 0].reshape(image.shape[0], image.shape[1], 1)
        a_channel = final_image[:, :, 0].reshape(image.shape[0], image.shape[1], 1)
        b_channel = final_image[:, :, 1].reshape(image.shape[0], image.shape[1], 1)

        final_image = np.concatenate((l_channel, a_channel, b_channel), axis=2)

        diff = image - final_image
        all_zeros = not np.any(diff)
        self.assertTrue(all_zeros)