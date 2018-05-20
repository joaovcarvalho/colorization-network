from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from loader import ColorfyImageLoader
from preprocessing import ColorfyPreprocessing
import os
import cv2
import sys
import numpy as np
from model import ColorfyModelFactory
from matplotlib import pyplot as plt
import datetime

directory = 'data/'
files = [f for (_, _, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

# input image dimensions
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols)

model = ColorfyModelFactory((img_rows, img_cols, 1)).get_model()
model.load_weights(sys.argv[1])

loader = ColorfyImageLoader(directory)
preprocessor = ColorfyPreprocessing(input_shape, cv2.COLOR_BGR2LAB)


data_generation = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True
    )

(x_train, _), (x_test, _) = cifar10.load_data()

# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

train_size = x_train.shape[0]
test_size = x_test.shape[0]


# def get_lab_image(x):
#     return cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
#
#
# x_train = [get_lab_image(image) for image in x_train]
# x_test = [get_lab_image(image) for image in x_test]

y_train = x_train
y_test = x_test

x_train = np.array(x_train).reshape(train_size, img_rows, img_cols, 3)
x_test = np.array(x_test).reshape(test_size, img_rows, img_cols, 3)

data_generation.fit(x_train)

choices = np.random.choice(x_test.shape[0], 10)
x_test = [ x for (index, x) in enumerate(x_test) if index in choices]

OUTPUT_SIZE = (64,64)

final_test_image = None

for original in x_test:
    original = original.reshape(32, 32, 3)
    x = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    x = np.array(x).reshape(1, img_rows, img_cols, 1).astype(float)

    x /= 255
    x -= 0.5

    result = model.predict(x)

    x = x.reshape(img_rows, img_cols, 1)

    color_space = result[0]

    a = color_space[:, :, 0]
    b = color_space[:, :, 1]

    a = a.reshape((img_rows, img_cols, 1))
    b = b.reshape((img_rows, img_cols, 1))

    colorized = np.concatenate((x, a, b), axis=2)
    colorized += .5
    colorized *= 255
    colorized = colorized.astype('uint8')
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = cv2.resize(colorized, OUTPUT_SIZE)

    x += 0.5
    x *= 255
    x = cv2.resize(x.astype('uint8'), OUTPUT_SIZE).reshape(OUTPUT_SIZE[0], OUTPUT_SIZE[1], 1)
    
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    
    original = cv2.resize(original, OUTPUT_SIZE).astype('uint8')

    result = np.append(x, colorized, axis=1)
    result = np.append(result, original, axis=1)

    if final_test_image is not None:
        final_test_image = np.append(final_test_image, result, axis=0)
    else:
        final_test_image = result
    # cv2.imshow('current', colorized)
    # cv2.waitKey(0)

import time
timestr = time.strftime("%Y%m%d_%H%M%S")

cv2.imwrite('results/results_{}.png'.format(timestr), final_test_image)
