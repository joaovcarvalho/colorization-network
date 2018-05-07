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
    featurewise_center=True,
    featurewise_std_normalization=True,
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


def get_lab_image(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2LAB)


x_train = [get_lab_image(image) for image in x_train]
x_test = [get_lab_image(image) for image in x_test]

y_train = x_train
y_test = x_test

x_train = np.array(x_train).reshape(train_size, img_rows, img_cols, 3)
x_test = np.array(x_test).reshape(test_size, img_rows, img_cols, 3)

data_generation.fit(x_train)

x_test = x_test[:20]
y_test = y_test[:20]

batches = 0
for x_batch, _ in data_generation.flow(x_test, y_test, batch_size=1):
    x = x_batch[0, :, :, 0]
    x = x.reshape(1, img_rows, img_cols, 1)

    result = model.predict(x)

    std = data_generation.std[0, 0]
    mean = data_generation.mean[0, 0]

    x = x * std[0] + mean[0]

    ab_space = result[0]

    a = ab_space[:, :, 0] * std[1] + mean[1]
    b = ab_space[:, :, 1] * std[2] + mean[2]

    a = a.reshape((img_rows, img_cols, 1))
    b = b.reshape((img_rows, img_cols, 1))

    gray = x[0]
    colorized = np.concatenate((gray, a, b), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = cv2.resize(colorized, (256, 256))

    gray = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray', gray)
    cv2.imshow('colorized', colorized)
    cv2.waitKey(0)

    batches += 1
    if batches >= len(x_test) / 1:
        # we need to break the loop by hand because
        # the generator loops indefinitely
        break

