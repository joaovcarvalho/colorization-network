import unittest

import numpy as np

from model import ColorfyModelFactory

TARGET_SIZE = (64, 64)


class ModelTest(unittest.TestCase):

    def test_soft_max(self):
        input_shape = TARGET_SIZE + (1,)
        model = ColorfyModelFactory(input_shape).get_model()
        input = np.zeros((1,) + input_shape)

        output = model.predict(input)
        sum_over_distribution = np.sum(output, axis=(1, 2))

        self.assertTrue(np.all(sum_over_distribution == 1.0))
