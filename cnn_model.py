from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.utils import Sequence
from keras.callbacks import Callback
import numpy as np
from cv2 import resize, imread
import os
import cv2

directory = 'data/'
files = [f for (dir, subdirs, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

# input image dimensions
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols)


def get_gray_image(image):
    l_channel, a_channel, b_channel = cv2.split(image)
    return l_channel


def get_color_image(image):
    l_channel, a_channel, b_channel = cv2.split(image)
    return cv2.merge([a_channel, b_channel])


def preprocess_image(file_name):
    image = imread(directory + file_name)
    if image is not None:
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return resize(lab_image, (128, 128))
    else:
        return None


class ImageNetSequence(Sequence):

    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        images = []

        i = 0
        while len(images) < self.batch_size:
            next_file_path = self.x[idx*self.batch_size + i]

            image = preprocess_image(next_file_path)
            if image is not None:
                images.append(image)
            i += 1

        gray_images = list(map(lambda x: get_gray_image(x), images))
        color_images = list(map(lambda x: get_color_image(x), images))

        gray_images, color_images = np.array(gray_images).reshape((self.batch_size, 128, 128, 1)), \
                                    np.array(color_images).reshape((self.batch_size, 128, 128, 2))
        return gray_images, color_images


class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1), padding="same"))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Dropout(0.25))
model.add(Conv2D(2, (3, 3), activation='relu', padding="same"))


# For a mean squared error regression problem
model.compile(optimizer='adam', loss='mse')

model.fit_generator(ImageNetSequence(files, 16), callbacks=[WeightsSaver(model, 5)])

model.save("colorfy")
