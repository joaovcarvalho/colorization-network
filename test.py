import sys
import math
import time

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
from plot_weights import plot_weights
from quantization import convert_quantization_to_image, convert_quantization_to_image_average
from visualize_activations import get_activation_model, plot_activations

img_rows = 128
img_cols = 128

input_shape = (img_rows, img_cols)

model = ColorfyModelFactory(input_shape + (1,)).get_model()
model.load_weights(sys.argv[1])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        )

train_generator = ColorizationDirectoryIterator(
        '../places-dataset',
        train_datagen,
        target_size=input_shape,
        batch_size=1,
        class_mode='original',
        )

OUTPUT_SIZE = (128, 128)

count = 0

HOW_MANY_IMAGES = 30

images_collected = []

DISPLAY_IMAGE = False
DISPLAY_DISTRIBUTION = False
SAVE_DISTRIBUTION = False
SAVE_TEST_IMAGES = True

SAVE_ACTIVATIONS = True

img_index = 0

activation_model = get_activation_model(model)


def get_image_from_network_result(result, l_channel, use_average=True):
    if use_average:
        color_space = convert_quantization_to_image_average(result, 16, 256, 3)
    else:
        color_space = convert_quantization_to_image(result, 16, 256)

    a = color_space[:, :, 0]
    b = color_space[:, :, 1]

    a = a.reshape((img_rows, img_cols, 1))
    b = b.reshape((img_rows, img_cols, 1))

    colorized = np.concatenate((l_channel, a, b), axis=2).astype('uint8')
    colorized = cv2.resize(colorized, OUTPUT_SIZE)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    return colorized


for input, y in train_generator:
    if count >= HOW_MANY_IMAGES:
        break
    count += 1

    batch_result = model.predict(input)

    if SAVE_ACTIVATIONS:
        activations = activation_model.predict(input)

    x = input.reshape(img_rows, img_cols, 1)

    prediction = batch_result[0]
    bins = 16

    x *= 7.61
    x += 119.85
    x = x.astype('uint8')

    constant_light = np.ones(x.shape) * 128

    color_map = get_image_from_network_result(prediction, constant_light)
    colorized = get_image_from_network_result(prediction, x)

    original = get_image_from_network_result(y[0], x, use_average=False)

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

    if SAVE_ACTIVATIONS:
        plot_activations(activations, model, count)

    if DISPLAY_DISTRIBUTION:
        plot_weights(current_sum / pixels_count)
    elif DISPLAY_IMAGE:
        cv2.waitKey(0)

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
    filename = 'results/results_{}.png'.format(timestr)
    cv2.imwrite(filename, final_test_image)

    print('Saved in: {}'.format(filename))
