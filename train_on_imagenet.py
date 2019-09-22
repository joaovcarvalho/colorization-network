import cv2
import keras
import numpy as np
from PIL import ImageFile
from keras.optimizers import Adam

ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.callbacks import TensorBoard, ReduceLROnPlateau
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

TARGET_SIZE = (64, 64)

NUM_EPOCHS = 10
BATCH_SIZE = 20
STEPS_PER_EPOCH = 230000
VALIDATION_STEPS = 1000
SAVE_MODEL_EVERY_N_BATCHES = 1000

model = ColorfyModelFactory(TARGET_SIZE + (1,)).get_model()
model.summary()

def get_loss_colorful(weights):
    def categorical_crossentropy_color(y_true, y_pred):

        # Flatten
        n, h, w, q = y_true.shape
        y_true = K.reshape(y_true, (n * h * w, q))
        y_pred = K.reshape(y_pred, (n * h * w, q))

        # multiply y_true by weights
        y_true = y_true * weights

        cross_ent = K.categorical_crossentropy(y_pred, y_true)
        cross_ent = K.mean(cross_ent, axis=-1)

        return cross_ent
    return categorical_crossentropy_color


def get_loss(weights):
    def colorize_loss(y_true, y_pred):
        diff = K.square(y_true - y_pred)
        prob_loss = K.sum(diff, axis=(0,1,2)) * weights
        return K.sum(prob_loss)
    return colorize_loss

optimizer = Adam(lr=0.00001)

model.compile(optimizer=optimizer, loss=get_loss(model.inputs[1]))

train_datagen = ImageDataGenerator(
        rescale=1./255,
)

data_folder = 'places/test/data/vision/torralba/deeplearning/images256'
# data_folder = 'data_256'

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
