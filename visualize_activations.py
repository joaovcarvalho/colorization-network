import numpy as np
from keras import models
from matplotlib import pyplot as plt
import cv2


def plot_activations(activations, model, index=0):
    layer_names = []
    for layer in model.layers[1:]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    bins = 16
    images_per_row = bins

    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        if 'softmax' not in layer_name:
            continue

        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        padding = 10
        grid_shape = (size * n_cols + n_cols * padding, images_per_row * size + images_per_row * padding, 3)
        display_grid = np.ones(grid_shape)
        display_grid[0] *= 256
        display_grid[1:] *= 127

        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_index = col * images_per_row + row
                division_factor = 256 / bins

                indexes = np.array([channel_index])

                a_channel = np.floor_divide(indexes, bins)
                b_channel = np.remainder(indexes, bins)

                a_channel = a_channel * float(division_factor)
                b_channel = b_channel * float(division_factor)

                a_channel = a_channel[0]
                b_channel = b_channel[0]

                channel_image = layer_activation[0, :, :, channel_index]
                a_channel = np.ones(channel_image.shape) * a_channel
                b_channel = np.ones(channel_image.shape) * b_channel

                channel_image[channel_image < 0.1] = 0.0
                a_channel[channel_image < 0.1] = 128
                b_channel[channel_image < 0.1] = 128

                final_image = np.zeros((channel_image.shape + (3,)))
                final_image[:, :, 0] = channel_image * 255
                final_image[:, :, 1] = a_channel
                final_image[:, :, 2] = b_channel

                x_padding = col * padding
                x_1 = col * size + x_padding
                x_2 = (col + 1) * size + x_padding

                y_padding = row * padding
                y_1 = row * size + y_padding
                y_2 = (row + 1) * size + y_padding

                display_grid[x_1: x_2, y_1: y_2] \
                    = final_image

        scale = 1. / (size*2)
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)

        display_grid = cv2.cvtColor(display_grid.astype('uint8'), cv2.COLOR_LAB2RGB)
        # plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.imshow(display_grid, aspect='auto')
        plt.savefig('activations/{}_{}'.format(layer_name, index))
        # plt.show()


def get_activation_model(model):
    layer_outputs = [layer.output for layer in model.layers[1:]]
    # Extracts the outputs of the top 1 layers
    return models.Model(inputs=model.input, outputs=layer_outputs)
