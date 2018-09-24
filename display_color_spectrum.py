import cv2

import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

# weights = np.load('weights.npy')
# # print(weights)
#
# x = np.arange(weights.shape[0])
#
# fig, ax = plt.subplots()
# plt.bar(x, weights)
# plt.show()
from quantization import convert_quantization_to_image
import time

BINS = 14
HOW_MANY_POSSIBLE_VALUES = BINS ** 2
HOW_MANY_L = 255

example_image = np.zeros((HOW_MANY_POSSIBLE_VALUES, HOW_MANY_L, HOW_MANY_POSSIBLE_VALUES))

for i in range(HOW_MANY_POSSIBLE_VALUES):
    for j in range(HOW_MANY_L):
        example_image[i, j, i] = 1.0

final_sum = np.zeros((BINS**2))
pixels_count = 0

color_space = convert_quantization_to_image(example_image, BINS, 256)
x = np.zeros((HOW_MANY_POSSIBLE_VALUES, HOW_MANY_L, 1))

for i in range(HOW_MANY_POSSIBLE_VALUES):
    for j in range(HOW_MANY_L):
        x[i, j, 0] = j * (255/HOW_MANY_L)

a = color_space[:, :, 0]
b = color_space[:, :, 1]

a = a.reshape((HOW_MANY_POSSIBLE_VALUES, HOW_MANY_L, 1))
b = b.reshape((HOW_MANY_POSSIBLE_VALUES, HOW_MANY_L, 1))

# for i in range(color_space.shape[0]):
#     for j in range(color_space.shape[1]):
#         a_pixel = a[i, j]
#         b_pixel = b[i, j]
#         print('A: {}, B: {}'.format(a_pixel, b_pixel))

print('A CHANNEL =========')
print(np.unique(a))
print('B CHANNEL =========')
print(np.unique(b))

colorized = np.concatenate((x, a, b), axis=2).astype('uint8')
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

print('B CHANNEL =========')
print(np.unique(colorized[:, :, 0]))
print('G CHANNEL =========')
print(np.unique(colorized[:, :, 1]))
print('R CHANNEL =========')
print(np.unique(colorized[:, :, 2]))

OUTPUT_SIZE = (800, 800)

colorized = cv2.resize(colorized, OUTPUT_SIZE)
cv2.imshow('test', colorized)
cv2.waitKey(0)