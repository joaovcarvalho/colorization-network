import math

import keras.backend as K
import numpy as np
import wandb
from PIL import ImageFile
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback

from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
# from models.models_colorful import load
from weights_saver_callback import WeightsSaverCallback

wandb.init(project="colorization")

K.clear_session()

ImageFile.LOAD_TRUNCATED_IMAGES = True

TARGET_SIZE = (64, 64)

NUM_EPOCHS = 10
BATCH_SIZE = 30
HOW_MANY_IMAGES = 2000000
STEPS_PER_EPOCH = math.floor(HOW_MANY_IMAGES / BATCH_SIZE)
VALIDATION_STEPS = 100
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

        squared_diff = K.abs(diff)
        distribution = K.sum(squared_diff, axis=(1, 2, 3))
        return K.mean(distribution)

    return loss_function


def categorical_crossentropy_color(y_true, y_pred):
    # Flatten
    shape = K.shape(y_true)
    n = shape[0]
    h = shape[1]
    w = shape[2]
    q = shape[3]

    y_true = K.reshape(y_true, (n * h * w, q))
    y_pred = K.reshape(y_pred, (n * h * w, q))

    cross_ent = K.categorical_crossentropy(y_pred, y_true)
    # cross_ent = K.mean(cross_ent, axis=-1)
    cross_ent = K.sum(cross_ent)

    return cross_ent


def colorize_loss(y_true, y_pred):
    diff = y_true - y_pred
    squared_diff = K.square(diff)
    distribution = K.sum(squared_diff)
    return distribution


LEARNING_RATE = 1e-5
optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.99)

model.compile(optimizer=optimizer, loss=colorize_loss)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

data_folder = 'places/data/vision/torralba/deeplearning/images256'
# data_folder = '../places-dataset'

data_generator = ColorizationDirectoryIterator(
    data_folder,
    train_datagen,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='original',
)


def schedule_policy(index, lr):
    if index == 0:
        return LEARNING_RATE

    return lr * 0.5


schedule_learning_rate = LearningRateScheduler(schedule_policy)

callbacks = [
    WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES),
    schedule_learning_rate,
    WandbCallback()
]

model.fit_generator(
    data_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=NUM_EPOCHS,
    validation_data=data_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    use_multiprocessing=True,
    workers=8
)
