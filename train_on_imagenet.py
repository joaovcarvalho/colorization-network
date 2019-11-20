import math
import numpy as np

import keras.backend as K
from PIL import ImageFile
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
# from models.models_colorful import load
from weights_saver_callback import WeightsSaverCallback

ImageFile.LOAD_TRUNCATED_IMAGES = True

TARGET_SIZE = (64, 64)

NUM_EPOCHS = 10
BATCH_SIZE = 30
HOW_MANY_IMAGES = 2000000
STEPS_PER_EPOCH = math.floor(HOW_MANY_IMAGES / BATCH_SIZE)
VALIDATION_STEPS = 1000
SAVE_MODEL_EVERY_N_BATCHES = 1000

INPUT_SHAPE = TARGET_SIZE + (1,)

model = ColorfyModelFactory(INPUT_SHAPE).get_model()

# NB_CLASSES = 256
# model = load(NB_CLASSES, INPUT_SHAPE, BATCH_SIZE)
model.summary()

weights_path = 'weights.npy'
weights = np.load(weights_path)

USE_WEIGHTS = False


def get_loss_with_weights(prior_distribution):
    def loss_function(y_true, y_pred):
        if USE_WEIGHTS:
            y_true = y_true * prior_distribution

        diff = y_true - y_pred
        squared_diff = K.square(diff)
        distribution = K.sum(squared_diff)
        return distribution

    return loss_function


def colorize_loss(y_true, y_pred):
    diff = y_true - y_pred
    squared_diff = K.square(diff)
    distribution = K.sum(squared_diff)
    return distribution


LEARNING_RATE = 1e-5
optimizer = Adam(lr=LEARNING_RATE)

model.compile(optimizer=optimizer, loss=get_loss_with_weights(weights))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

data_folder = 'places/data/vision/torralba/deeplearning/images256'
# data_folder = 'imagenet'

data_generator = ColorizationDirectoryIterator(
    data_folder,
    train_datagen,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='original',
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=0.000001)

callbacks = [WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES), reduce_lr]

model.fit_generator(
    data_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=NUM_EPOCHS,
    validation_data=data_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
)
