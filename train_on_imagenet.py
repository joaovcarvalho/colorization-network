import cv2
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
from weights_saver_callback import WeightsSaverCallback

TARGET_SIZE = (32, 32)

NUM_EPOCHS = 10
BATCH_SIZE = 32
STEPS_PER_EPOCH = 100000
VALIDATION_STEPS = 2000
SAVE_MODEL_EVERY_N_BATCHES = 10

model = ColorfyModelFactory(TARGET_SIZE + (1,)).get_model()
model.summary()

# For a mean squared error regression problem
model.compile(optimizer='adam', loss='mse')

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

callbacks = [WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES)]

model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks
)
