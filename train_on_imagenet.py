import cv2
import keras
import numpy as np
from PIL import ImageFile
from keras.optimizers import Adam

ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.callbacks import TensorBoard
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
from weights_saver_callback import WeightsSaverCallback

TARGET_SIZE = (128, 128)

NUM_EPOCHS = 5
BATCH_SIZE = 4
STEPS_PER_EPOCH = 50000
VALIDATION_STEPS = 10000
SAVE_MODEL_EVERY_N_BATCHES = 500

model = ColorfyModelFactory(TARGET_SIZE + (1,)).get_model()
model.summary()

weights = np.load('weights.npy')
weights = 1 - weights
weights_v = K.constant(weights)


def colorize_loss(y_true, y_pred):
    global weights_v
    mult = y_true - y_pred
    square = K.square(mult)
    sum = K.sum(square, axis=(1, 2))
    weighted_sum = sum * weights_v
    return K.sum(weighted_sum)


optimizer = Adam(lr=0.0001)

model.compile(optimizer=optimizer, loss=colorize_loss)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

train_generator = ColorizationDirectoryIterator(
        'imagenet',
        train_datagen,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='original',
)

validation_generator = ColorizationDirectoryIterator(
        'imagenet',
        test_datagen,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='original',
)

tbCallBack = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

callbacks = [WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES), tbCallBack]

model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
)
