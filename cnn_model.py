from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from image_path_sequence import ImagesPathSequence
# from loader import ColorfyImageLoader
from weights_saver_callback import WeightsSaverCallback
from keras.callbacks import TensorBoard
from model import ColorfyModelFactory
from preprocessing import ColorfyPreprocessing
import os
import cv2
import numpy as np

directory = 'data/'
files = [f for (_, _, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

# input image dimensions
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols)

# loader = ColorfyImageLoader(directory)
preprocessor = ColorfyPreprocessing(input_shape, cv2.COLOR_BGR2LAB)

model = ColorfyModelFactory((img_rows, img_cols, 1)).get_model()

# For a mean squared error regression problem
model.compile(optimizer='adam', loss='mse')

# data_generation = ImagesPathSequence(files, 16, loader, preprocessor)

data_generation = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
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


# def get_lab_image(x):
#     return cv2.cvtColor(x, cv2.COLOR_RGB2LAB)

# x_train = [get_lab_image(image) for image in x_train]
# x_test = [get_lab_image(image) for image in x_test]

y_train = x_train
y_test = x_test

x_train = np.array(x_train).reshape(train_size, img_rows, img_cols, 3)
x_test = np.array(x_test).reshape(test_size, img_rows, img_cols, 3)

data_generation.fit(x_train)

# history = model.fit_generator(
#     data_generation.flow(x_train, y_train, batch_size=32),
#     callbacks=[WeightsSaverCallback(model, every=10)],
#     epochs=1
# )

NUM_EPOCHS = 10
BATCH_SIZE = 128
SAVE_MODEL_EVERY_N_BATCHES = 10

tbCallBack = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# here's a more "manual" example
for e in range(NUM_EPOCHS):
    print('Epoch {}/{}'.format(e, NUM_EPOCHS))
    batches = 0
    for x_batch, _ in data_generation.flow(x_train, y_train, batch_size=BATCH_SIZE):

        x = []

        for image in x_batch:
            x += [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]

        x = np.array(x).reshape(BATCH_SIZE, img_rows, img_cols, 1)
        y = x_batch

        model.fit(x, y, callbacks=[WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES), tbCallBack])
        print "Batch {0} / {1} ".format(batches, len(x_train) / BATCH_SIZE)
        batches += 1
        if batches >= len(x_train) / BATCH_SIZE:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

# summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('learning_curve.png')
#
# model.save("colorfy")
