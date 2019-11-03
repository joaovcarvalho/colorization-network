import sys
import math
import time

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
from plot_weights import plot_weights
from quantization import convert_quantization_to_image

img_rows = 128
img_cols = 128

input_shape = (img_rows, img_cols)

model = ColorfyModelFactory(input_shape + (1,)).get_model()
model.load_weights(sys.argv[1])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        )

train_generator = ColorizationDirectoryIterator(
        'imagenet',
        train_datagen,
        target_size=input_shape,
        batch_size=1,
        class_mode='original'
        )

OUTPUT_SIZE = (300, 300)

count = 0

HOW_MANY_IMAGES = 100

images_collected = []

DISPLAY_IMAGE = True
DISPLAY_DISTRIBUTION = False
SAVE_DISTRIBUTION = False
SAVE_TEST_IMAGES = False

img_index = 0

for x, y in train_generator:
    if count >= HOW_MANY_IMAGES:
        break
    count += 1

    batch_result = model.predict(x)

    x = x.reshape(img_rows, img_cols, 1)

    prediction = batch_result[0]
    bins = 16
    color_space = convert_quantization_to_image(prediction, bins, 256)

    a = color_space[:, :, 0]
    b = color_space[:, :, 1]

    a = a.reshape((img_rows, img_cols, 1))
    b = b.reshape((img_rows, img_cols, 1))

    x *= 255
    x = x.astype('uint8')

    constant_light = np.ones(a.shape) * 128

    color_map = np.concatenate((constant_light, a, b), axis=2).astype('uint8')
    color_map = cv2.resize(color_map, OUTPUT_SIZE)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_LAB2BGR)

    colorized = np.concatenate((x, a, b), axis=2).astype('uint8')
    colorized = cv2.resize(colorized, OUTPUT_SIZE)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

    original = convert_quantization_to_image(y[0], bins, 256)

    a_original = original[:, :, 0].reshape((img_rows, img_cols, 1))
    b_original = original[:, :, 1].reshape((img_rows, img_cols, 1))

    original = np.concatenate((x, a_original, b_original), axis=2).astype('uint8')
    original = cv2.resize(original, OUTPUT_SIZE)
    original = cv2.cvtColor(original, cv2.COLOR_LAB2BGR)

    original = cv2.resize(original, OUTPUT_SIZE).astype('uint8')

    x = cv2.resize(x, OUTPUT_SIZE).reshape(OUTPUT_SIZE[0], OUTPUT_SIZE[1], 1)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    result = np.append(color_map, colorized, axis=1)
    result = np.append(result, original, axis=1)
    images_collected.append(result)

    current_sum = np.sum(batch_result, axis=(0, 1, 2))
    assert current_sum.shape == (bins ** 2,)
    pixels_count = batch_result.shape[0] * batch_result.shape[1] * batch_result.shape[2]

    if DISPLAY_IMAGE:
        cv2.imshow('result', result)

    if DISPLAY_DISTRIBUTION:
        plot_weights(current_sum / pixels_count)
    else:
        cv2.waitKey(2000)

    if SAVE_DISTRIBUTION:
        plot_weights(current_sum / pixels_count, 'distributions/distribution_{}'.format(img_index))

    img_index += 1

if SAVE_TEST_IMAGES:
    timestr = time.strftime("%Y%m%d_%H%M%S")

    final_test_image = None
    rows = math.ceil(len(images_collected) / 3)

    rows_images = []

    current_image = None
    for i, image in enumerate(images_collected):
        if current_image is not None:
            current_image = np.append(current_image, image, axis=1)
        else:
            current_image = image

        if i % 3 == 2:
            rows_images.append(current_image)
            current_image = None

    for row_image in rows_images:
        print(row_image.shape)
        if final_test_image is not None:
            print(final_test_image.shape)
            final_test_image = np.append(final_test_image, row_image, axis=0)
        else:
            final_test_image = row_image

    # cv2.imshow('test', final_test_image)
    # cv2.waitKey(0)
    # cv2.imwrite('results/results_{}.png'.format(timestr), final_test_image)
