import numpy as np
import matplotlib.pyplot as plt
import cv2
from quantization import convert_quantization_to_image

weights = np.load('weights.npy')

x = np.arange(weights.shape[0])
fig, ax = plt.subplots()
bar = plt.bar(x, weights)

BINS = 16

for i in range(weights.shape[0]):
    pixel_distribution = np.zeros((1, 1, BINS**2))
    pixel_distribution[0, 0, i] = 1.0
    color_space = convert_quantization_to_image(pixel_distribution, BINS, 255).astype('uint8')

    a = color_space[:, :, 0].reshape((1, 1, 1))
    b = color_space[:, :, 1].reshape((1, 1, 1))

    colorized = np.concatenate((np.ones((1, 1, 1)) * 128, a, b), axis=2).astype('uint8')
    rgb_color = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR) / 255.0
    rgb_tuple = tuple(rgb_color[0, 0, :])
    bar[i].set_color(rgb_tuple)

plt.show()
