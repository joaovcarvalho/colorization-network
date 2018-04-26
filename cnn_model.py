from model import ColorfyModelFactory
from keras.utils import Sequence
from keras.callbacks import Callback
import numpy as np
from preprocessing import ColorfyPreprocessing
import os
import cv2

directory = 'data/'
files = [f for (_, _, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

# input image dimensions
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols)

preprocessor = ColorfyPreprocessing(directory, input_shape, cv2.COLOR_BGR2LAB)


class ImageNetSequence(Sequence):

    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        images = []
        gray_images = []
        color_images = []

        i = 0
        while len(images) < self.batch_size:
            next_file_path = self.x[idx*self.batch_size + i]

            image = preprocessor.process(next_file_path)
            if image is not None:
                images.append(image)
                gray_images.append(preprocessor.get_gray_image())
                color_images.append(preprocessor.get_color_image())
            i += 1

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
            name = 'weights/weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1


model = ColorfyModelFactory((128, 128, 1)).get_model()

# For a mean squared error regression problem
model.compile(optimizer='adam', loss='mse')

model.fit_generator(ImageNetSequence(files, 16), callbacks=[WeightsSaver(model, 5)])

model.save("colorfy")
