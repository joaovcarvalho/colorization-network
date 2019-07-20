import math
import sys

import tensorflow as tf
from PIL import ImageFile
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from losses import colorful_colorization_loss as loss_function
from model import ColorfyModelFactory
from weights_saver_callback import WeightsSaverCallback

ImageFile.LOAD_TRUNCATED_IMAGES = True

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

TARGET_SIZE = (128, 128)

HOW_MANY_IMAGES_PER_EPOCH = 2.3e6
NUM_EPOCHS = 4
BATCH_SIZE = 10
STEPS_PER_EPOCH = HOW_MANY_IMAGES_PER_EPOCH / BATCH_SIZE
SAVE_MODEL_EVERY_N_BATCHES = 500

model = ColorfyModelFactory(TARGET_SIZE + (1,)).get_model()

if len(sys.argv) > 1:
    model.load_weights(sys.argv[1])

model.summary()

# Parameters extracted from Colorful Image Colorization paper
initial_learning_rate = 1 / 10e4
optimizer = Adam(lr=initial_learning_rate, beta_1=.9, beta_2=.99, decay=.001)


def step_decay(epoch):
    drop = 0.1
    epochs_drop = 2.0
    return initial_learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))


model.compile(optimizer=optimizer, loss=loss_function)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

data_folder = 'places/test/data/vision/torralba/deeplearning/images256'

data_generator = ColorizationDirectoryIterator(
    data_folder,
    train_datagen,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='original',
)

VALIDATION_STEPS = 1000
VALIDATION_BATCH_SIZE = 10

validation_generator = ColorizationDirectoryIterator(
    data_folder,
    train_datagen,
    target_size=TARGET_SIZE,
    batch_size=VALIDATION_BATCH_SIZE,
    class_mode='original',
)

# tensor_board_callback = TensorBoard(
#     log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True, write_grads=True
# )

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1)


lrate = LearningRateScheduler(step_decay)

callbacks = [WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES), reduce_lr, lrate]

model.fit_generator(
    data_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
)
