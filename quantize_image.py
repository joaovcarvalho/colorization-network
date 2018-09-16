import cv2

import numpy as np
from quantization import convert_quantization_to_image, quantize_lab_image
import sys

BINS = 16
image_type = 'uint8'

original = cv2.imread(sys.argv[1])
image = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

l_channel = image[:, :, 0]
l_channel = l_channel.reshape((image.shape[0], image.shape[1], 1))

quantum = quantize_lab_image(image, bins=BINS, max_value=255)
color_space = convert_quantization_to_image(quantum, BINS, 255)

a = color_space[:, :, 0]
b = color_space[:, :, 1]

a = a.reshape((image.shape[0], image.shape[1], 1)).astype(image_type)
b = b.reshape((image.shape[0], image.shape[1], 1)).astype(image_type)

l_channel = l_channel.astype(image_type)

colorized = np.concatenate((l_channel, a, b), axis=2).astype(image_type)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

OUTPUT_SIZE = (800, 800)

colorized = cv2.resize(colorized, OUTPUT_SIZE).astype('uint8')
original = cv2.resize(original, OUTPUT_SIZE).astype('uint8')

final_test_image = np.append(colorized, original, axis=1)
cv2.imshow('test', final_test_image)
cv2.waitKey(0)