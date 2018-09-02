from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from loader import ColorfyImageLoader
from preprocessing import ColorfyPreprocessing
import os
import cv2
import sys
import time
import numpy as np
from autoencoder_model import AutoEncoderFactory
from utils import save_timestamped_result

directory = 'data/'
files = [f for (_, _, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

# input image dimensions
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols)

model = AutoEncoderFactory((img_rows, img_cols, 3)).get_model()
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

train_size = x_train.shape[0]
test_size = x_test.shape[0]

y_train = x_train
y_test = x_test

x_train = np.array(x_train).reshape(train_size, img_rows, img_cols, 3)
x_test = np.array(x_test).reshape(test_size, img_rows, img_cols, 3)

data_generation.fit(x_train)

choices = np.random.choice(x_test.shape[0], 10)
x_test = [x for (index, x) in enumerate(x_test) if index in choices]

OUTPUT_SIZE = (64, 64)

final_test_image = None

for original in x_test:
    original = original.reshape(32, 32, 3)
    x = np.array(original).reshape(1, img_rows, img_cols, 3).astype(float)

    x /= 255
    x -= 0.5

    result = model.predict(x)

    x = x.reshape(img_rows, img_cols, 3)

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
    colorized *= 255
    colorized = colorized.astype('uint8')
    colorized = cv2.resize(colorized, OUTPUT_SIZE)

    x += 0.5
    x *= 255
    x = cv2.resize(x.astype('uint8'), OUTPUT_SIZE).reshape(OUTPUT_SIZE[0], OUTPUT_SIZE[1], 3)

    original = cv2.resize(original, OUTPUT_SIZE).astype('uint8')

    result = np.append(x, colorized, axis=1)
    result = np.append(result, original, axis=1)

    if final_test_image is not None:
        final_test_image = np.append(final_test_image, result, axis=0)
    else:
        final_test_image = result


save_timestamped_result(folder='autoencoder', image=final_test_image)
