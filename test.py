import os
import sys

import cv2
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from loader import ColorfyImageLoader
from model import ColorfyModelFactory
from preprocessing import ColorfyPreprocessing
from quantization import convert_quantization_to_image

directory = 'data/'
files = [f for (_, _, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

# input image dimensions
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols)

model = ColorfyModelFactory((32, 32, 1)).get_model()
model.load_weights(sys.argv[1])

loader = ColorfyImageLoader(directory)
preprocessor = ColorfyPreprocessing(input_shape, cv2.COLOR_BGR2LAB)

data_generation = ImageDataGenerator()

(x_train, _), (x_test, _) = cifar10.load_data()

train_size = x_train.shape[0]
test_size = x_test.shape[0]

y_train = x_train
y_test = x_test

x_train = np.array(x_train).reshape(train_size, img_rows, img_cols, 3)
x_test = np.array(x_test).reshape(test_size, img_rows, img_cols, 3)

data_generation.fit(x_train)

choices = np.random.choice(x_test.shape[0], 10)
x_test = [x for (index, x) in enumerate(x_test) if index in choices]

OUTPUT_SIZE = (32, 32)

final_test_image = None

img_cols = 32
img_rows = 32
input_shape = (32, 32)

for original in x_test:
    original = original.reshape(32, 32, 3)
    x = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(x, input_shape)
    x = np.array(x).reshape(1, img_rows, img_cols, 1).astype(float)

    x /= 255

    result = model.predict(x)

    x = x.reshape(img_rows, img_cols, 1)

    color_space = convert_quantization_to_image(result[0], 12, 255)

    a = color_space[:, :, 0]
    b = color_space[:, :, 1]

    a = a.reshape((img_rows, img_cols, 1))
    b = b.reshape((img_rows, img_cols, 1))

    x *= 255

    colorized = np.concatenate((x, a, b), axis=2)
    colorized = cv2.resize(colorized, OUTPUT_SIZE)

    x = cv2.resize(x.astype('uint8'), OUTPUT_SIZE).reshape(OUTPUT_SIZE[0], OUTPUT_SIZE[1], 1)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    original = cv2.resize(original, OUTPUT_SIZE).astype('uint8')

    result = np.append(x, colorized, axis=1)
    result = np.append(result, original, axis=1)

    if final_test_image is not None:
        final_test_image = np.append(final_test_image, result, axis=0)
    else:
        final_test_image = result

import time

timestr = time.strftime("%Y%m%d_%H%M%S")

# cv2.imshow('test', final_test_image)
# cv2.waitKey(0)
cv2.imwrite('results/results_{}.png'.format(timestr), final_test_image)
