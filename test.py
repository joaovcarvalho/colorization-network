import sys
from collections import namedtuple

import matplotlib.pyplot as plt

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from aggregate_images import save_aggregate_images
from compare import compare_lab_images
from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
from plot_weights import plot_weights
from post_processing import get_image_from_network_result
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
        seed=3,
        )

OUTPUT_SIZE = (128, 128)

count = 0

HOW_MANY_IMAGES = 1000

images_collected = []

DISPLAY_IMAGE = True
DISPLAY_DISTRIBUTION = False
SAVE_DISTRIBUTION = False
SAVE_TEST_IMAGES = False

SAVE_ACTIVATIONS = False

img_index = 0

activation_model = get_activation_model(model)

def plot_graph(data):
    plt.subplot(2, 1, 2)
    x = range(len(data))
    plt.ylim(0, 100)
    plt.plot(x, data)
    plt.xlabel('Limite(t)')
    plt.ylabel('Acuracia(%)')

all_final_accuracies = []


for input, output in train_generator:
    if count >= HOW_MANY_IMAGES:
        break
    count += 1

    batch_result = model.predict(input)

    if SAVE_ACTIVATIONS:
        activations = activation_model.predict(input)
        plot_activations(activations, model, count)

    l_channel = input.reshape(img_rows, img_cols, 1)

    prediction = batch_result[0]

    h, w, nb_q = prediction.shape
    
    bins = 16

    l_channel *= 7.61
    l_channel += 119.85
    l_channel = l_channel.astype('uint8')

    constant_light = np.ones(l_channel.shape) * 128

    colorized, lab_image = get_image_from_network_result(
        prediction,
        l_channel,
        (img_rows, img_cols),
        OUTPUT_SIZE,
    )
    original, lab_original = get_image_from_network_result(
        output[0],
        l_channel,
        (img_rows, img_cols),
        OUTPUT_SIZE,
        use_average=False
    )

    l_channel = cv2.resize(l_channel, OUTPUT_SIZE).reshape(OUTPUT_SIZE[0], OUTPUT_SIZE[1], 1)
    l_channel = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2BGR)

    result = np.append(colorized, original, axis=1)
    images_collected.append(result)

    current_sum = np.sum(batch_result, axis=(0, 1, 2))
    assert current_sum.shape == (bins ** 2,)
    pixels_count = batch_result.shape[0] * batch_result.shape[1] * batch_result.shape[2]

    if DISPLAY_IMAGE:
        plt.subplot(2, 1, 1)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result)
        # cv2.imshow('result', result)
    
    if DISPLAY_DISTRIBUTION:
        plot_weights(current_sum / pixels_count)
    # elif DISPLAY_IMAGE:
    #     cv2.waitKey(1000)

    cumulative_accuracy = [
        compare_lab_images(lab_image, lab_original, i) * 100
        for i in range(150)
    ]

    area_under_curve = sum(cumulative_accuracy) / (150 * 100)
    print('AUC: ', area_under_curve)

    all_final_accuracies += [sum(cumulative_accuracy)]

    print('#{}'.format(count))
    plot_graph(cumulative_accuracy)
    fig_name = 'accuracy/{:.2f}_{}.png'.format(area_under_curve * 100, count)
    print('Saving fig {}'.format(fig_name))
    plt.savefig(fig_name)
    plt.clf()
    # plt.show()

    if SAVE_DISTRIBUTION:
        plot_weights(current_sum / pixels_count, 'distributions/distribution_{}'.format(img_index))

    img_index += 1

plot_graph(all_final_accuracies)
plt.show()

print('Accuracy final', sum(all_final_accuracies) / (len(all_final_accuracies) * 150 * 100))

if SAVE_TEST_IMAGES:
    save_aggregate_images(images_collected)
