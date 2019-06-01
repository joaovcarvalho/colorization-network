import sys

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
from quantization import convert_quantization_to_image
from visualizations import get_activation_model

img_rows = 128
img_cols = 128

input_shape = (img_rows, img_cols)

model = ColorfyModelFactory(input_shape + (1,)).get_model()
model.load_weights(sys.argv[1])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

train_generator = ColorizationDirectoryIterator(
    'imagenet',
    train_datagen,
    target_size=input_shape,
    batch_size=1,
    class_mode='original'
)

# model.compile(Adam(),loss=colorize_loss, metrics=['accuracy'])
# loss, accuracy = model.evaluate_generator(train_generator,
#                                           workers=3,
#                                           use_multiprocessing=True)
#
# print('Accuracy {}'.format(accuracy * 100))

OUTPUT_SIZE = (400, 400)
final_test_image = None

count = 0

activation_model = get_activation_model(model)

FINAL_IMAGE_FORMAT = cv2.COLOR_LAB2BGR
HOW_MANY_TO_PLOT = 20

for x, y in train_generator:
    if count >= HOW_MANY_TO_PLOT:
        break
    count += 1

    result = model.predict(x)
    activations = activation_model.predict(x)

    x = x.reshape(img_rows, img_cols, 1)

    color_space = convert_quantization_to_image(result[0], 16, 255)

    a = color_space[:, :, 0]
    b = color_space[:, :, 1]

    a = a.reshape((img_rows, img_cols, 1))
    b = b.reshape((img_rows, img_cols, 1))

    x += 50
    x = x.astype('uint8')

    constant_light = np.ones(a.shape) * 255
    colorized = np.concatenate((x, a, b), axis=2).astype('uint8')
    colorized = cv2.resize(colorized, OUTPUT_SIZE)
    colorized = cv2.cvtColor(colorized, FINAL_IMAGE_FORMAT)

    original = convert_quantization_to_image(y[0], 16, 255)

    a_original = original[:, :, 0].reshape((img_rows, img_cols, 1))
    b_original = original[:, :, 1].reshape((img_rows, img_cols, 1))

    original = np.concatenate((x, a_original, b_original), axis=2).astype('uint8')
    original = cv2.resize(original, OUTPUT_SIZE)
    original = cv2.cvtColor(original, FINAL_IMAGE_FORMAT)

    original = cv2.resize(original, OUTPUT_SIZE).astype('uint8')

    x = cv2.resize(x, OUTPUT_SIZE).reshape(OUTPUT_SIZE[0], OUTPUT_SIZE[1], 1)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    result = np.append(x, colorized, axis=1)
    result = np.append(result, original, axis=1)

    cv2.imshow('result', result)
    cv2.waitKey(500)
    # plot_activations(activations, model)
