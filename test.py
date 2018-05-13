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

choices = np.random.choice(x_test.shape[0], 20)
x_test = [ x for (index, x) in enumerate(x_test) if index in choices]

for x in x_test:
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = np.array(x).reshape(1, img_rows, img_cols, 1).astype(float)

    x /= 255
    x -= 0.5

    result = model.predict(x)

    x = x.reshape(img_rows, img_cols, 1)

    color_space = result[0]

    b = color_space[:, :, 0]
    g = color_space[:, :, 1]
    r = color_space[:, :, 2]

    b += 0.5
    g += 0.5
    r += 0.5

    r = r.reshape((img_rows, img_cols, 1))
    g = g.reshape((img_rows, img_cols, 1))
    b = b.reshape((img_rows, img_cols, 1))

    colorized = np.concatenate((b, g, r), axis=2)
    colorized = cv2.resize(colorized, (256, 256))

    x += 0.5
    x = cv2.resize(x, (256, 256))

    cv2.imshow('gray', x)
    cv2.imshow('colorized', colorized)
    cv2.waitKey(0)