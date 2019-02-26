import os

import cv2
import numpy as np
from keras.callbacks import TensorBoard
from keras.datasets import cifar10

from model import ColorfyModelFactory
# from loader import ColorfyImageLoader
from weights_saver_callback import WeightsSaverCallback

directory = 'data/'
files = [f for (_, _, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

# input image dimensions
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols)

# loader = ColorfyImageLoader(directory)
# preprocessor = ColorfyPreprocessing(input_shape, cv2.COLOR_BGR2LAB)

model = ColorfyModelFactory((img_rows, img_cols, 1)).get_model()

# For a mean squared error regression problem
model.compile(optimizer='adam', loss='mse')

(x_train, _), (x_test, _) = cifar10.load_data()

train_size = x_train.shape[0]
test_size = x_test.shape[0]

y_train = x_train
y_test = x_test

x_train = np.array(x_train).reshape(train_size, img_rows, img_cols, 3)
x_test = np.array(x_test).reshape(test_size, img_rows, img_cols, 3)

NUM_EPOCHS = 10
BATCH_SIZE = 128
SAVE_MODEL_EVERY_N_BATCHES = 10

tbCallBack = TensorBoard(log_dir='./graph',
                         batch_size=BATCH_SIZE,
                         histogram_freq=2,
                         write_graph=True,
                         write_images=True,
                         write_grads=True)

callbacks = [WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES), tbCallBack]

gray_images = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_train]).reshape(train_size, img_rows, img_cols,
                                                                                       1).astype(float)
color_images = x_train.astype(float)

gray_images /= 255
color_images /= 255

gray_images -= .5
color_images -= .5

history = model.fit(gray_images, color_images,
                    validation_split=0.1,
                    epochs=NUM_EPOCHS,
                    callbacks=callbacks,
                    batch_size=BATCH_SIZE)

